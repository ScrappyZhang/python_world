# 10_ElasticSearch dis_max实现best fields策略进行多字段搜索

### 概述

- 基于多个 field 查询如 title(标题) content 内容.
  搜索title或content中包含java或solution的帖子
  期望：如果title中包含了java和solution 。或者 content 中保护 java和solution 这样的doc 优先排在前面。
- best fields策略，就是说，搜索到的结果，应该是某一个field中匹配到了尽可能多的关键词，被排在前面；而不是尽可能多的field匹配到了少数的关键词，排在了前面。

### 例子：

#### 更新title字段

```
POST /forum/article/_bulk
{ "update": { "_id": "1"} }
{ "doc" : {"title" : "this is java and elasticsearch blog"} }
{ "update": { "_id": "2"} }
{ "doc" : {"title" : "this is java blog"} }
{ "update": { "_id": "3"} }
{ "doc" : {"title" : "this is elasticsearch blog"} }
{ "update": { "_id": "4"} }
{ "doc" : {"title" : "this is java, elasticsearch, hadoop blog"} }
{ "update": { "_id": "5"} }
{ "doc" : {"title" : "this is spark blog"} }
```

#### 为帖子数据增加content字段

```
POST /forum/article/_bulk
{ "update": { "_id": "1"} }
{ "doc" : {"content" : "i like to write best elasticsearch article"} }
{ "update": { "_id": "2"} }
{ "doc" : {"content" : "i think java is the best programming language"} }
{ "update": { "_id": "3"} }
{ "doc" : {"content" : "i am only an elasticsearch beginner"} }
{ "update": { "_id": "4"} }
{ "doc" : {"content" : "elasticsearch and hadoop are all very good solution, i am a beginner"} }
{ "update": { "_id": "5"} }
{ "doc" : {"content" : "spark is best big data solution based on scala ,an programming language similar to java"} }
```

### 搜索title或content中包含java或solution的帖子

```
GET /forum/article/_search
{
    "query": {
        "bool": {
            "should": [
                { "match": { "title": "java solution" }},
                { "match": { "content":  "java solution" }}
            ]
        }
    }
}
```

- 期望的是doc5，结果是doc2, doc4排在了前面 (doc 5 中 content字段 中保护了 java 和 solution)

- 计算每个document的relevance score：每个query的分数，乘以matched query数量，除以总query数量

- 算一下doc4的分数

  > { "match": { "title": "java solution" }}，针对doc4，是有一个分数的
  > { "match": { "content":  "java solution" }}，针对doc4，也是有一个分数的
  >
  > 所以是两个分数加起来，比如说，1.1 + 1.2 = 2.3
  > matched query数量 = 2
  > 总query数量 = 2
  >
  > 2.3 * 2 / 2 = 2.3

- 算一下doc5的分数

  > { "match": { "title": "java solution" }}，针对doc5，是没有分数的
  > { "match": { "content":  "java solution" }}，针对doc5，是有一个分数的
  >
  > 所以说，只有一个query是有分数的，比如2.3
  > matched query数量 = 1
  > 总query数量 = 2
  >
  > 2.3 * 1 / 2 = 1.15
  >
  > doc5的分数 = 1.15 < doc4的分数 = 2.3

### 5、best fields策略，dis_max

best fields策略，就是说，搜索到的结果，应该是某一个field中匹配到了尽可能多的关键词，被排在前面；而不是尽可能多的field匹配到了少数的关键词，排在了前面

dis_max语法，直接取多个query中，分数最高的那一个query的分数即可

> { "match": { "title": "java solution" }}，针对doc4，是有一个分数的，1.1
> { "match": { "content":  "java solution" }}，针对doc4，也是有一个分数的，1.2
> 取最大分数，1.2
> { "match": { "title": "java solution" }}，针对doc5，是没有分数的
> { "match": { "content":  "java solution" }}，针对doc5，是有一个分数的，2.3
>
> 取最大分数，2.3
>
> 然后doc4的分数 = 1.2 < doc5的分数 = 2.3，所以doc5就可以排在更前面的地方，符合我们的需要

```
GET /forum/article/_search
{
    "query": {
        "dis_max": {
            "queries": [
                { "match": { "title": "java solution" }},
                { "match": { "content":  "java solution" }}
            ]
        }
    }
}
```

## 基于tie_breaker参数优化dis_max搜索效果

dis_max，只是取分数最高的那个query的分数而已。
可能在实际场景中出现的一个情况是这样的：

1、某个帖子，doc1，title中包含java（1），content不包含java beginner任何一个关键词
2、某个帖子，doc2，content中包含beginner（1），title中不包含任何一个关键词
3、某个帖子，doc3，title中包含java（1），content中包含beginner（1）
4、以上3个doc的最高score都是1所有最终出来的排序不一定是想要的结果
5、最终搜索，可能出来的结果是，doc1和doc2排在doc3的前面，而不是我们期望的doc3排在最前面
原因：

**dis_max只取某一个query最大的分数，完全不考虑其他query的分数**

### 搜索title或content中包含java beginner的帖子

```
GET /forum/article/_search
{
    "query": {
        "dis_max": {
            "queries": [
                { "match": { "title": "java blog" }},
                { "match": { "content":  "java blog" }}
            ]
        }
    }
}
```

### 使用tie_breaker将其他query的分数也考虑进去

tie_breaker参数的意义，在于说，将其他query的分数，乘以tie_breaker，然后综合与最高分数的那个query的分数，综合在一起进行计算
除了取最高分以外，还会考虑其他的query的分数

tie_breaker的值，在0~1之间，是个小数，就ok

```
GET /forum/article/_search
{
    "query": {
        "dis_max": {
            "queries": [
                { "match": { "title": "java blog" }},
                { "match": { "content":  "java blog" }}
            ],
            "tie_breaker": 0.3
        }
    }
}

```

## multi_match语法实现dis_max+tie_breaker

**dis_max**

score沿用子查询score的最大值
**tie_breaker**

可以通过tie_breaker来控制其他field的得分
**minimum_should_match**，主要是作用:

1、去长尾，long tail
2、长尾，比如你搜索5个关键词，但是很多结果是只匹配1个关键词的，其实跟你想要的结果相差甚远，这些结果就是长尾

3、minimum_should_match，控制搜索结果的精准度，只有匹配一定数量的关键词的数据，才能返回



```
GET /forum/article/_search?explain=true
{
  "query": {
    "multi_match": {
        "query":                "java blog",
        "type":                 "best_fields", 
        "fields":               [ "title^2", "content" ],
        "tie_breaker":          0.3,
        "minimum_should_match": "50%" 
    }
  } 
}
```

等同于：

```
GET /forum/article/_search
{
  "query": {
    "dis_max": {
      "queries":  [
        {
          "match": {
            "title": {
              "query": "java blog",
              "minimum_should_match": "50%",
			  "boost": 2
            }
          }
        },
        {
          "match": {
            "content": {
              "query": "java blog",
              "minimum_should_match": "50%"
            }
          }
        }
      ],
      "tie_breaker": 0.3
    }
  } 
}
```

