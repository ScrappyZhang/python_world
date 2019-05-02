# 4_ElaticSearch 使用terms搜索多个值

## 概述

- es 中如何实现 sql 中的in，使用terms实现

- 语法如下：


```
term: {"field": "value"}
terms: {"field": ["value1", "value2"]}
```

- 
  sql中的in：

```
select * from tbl where col in ("value1", "value2")
```

### 1、为帖子数据增加tag字段

- 为文章添加tag标签 其中id = i tag 为 java和hadoop.
- id= 2 tag为 java
- id= 3的 tag 为 hadoop
- id=4 的 tag 为 java elasticsearch

```
POST /forum/article/_bulk
{ "update": { "_id": "1"} }
{ "doc" : {"tag" : ["java", "hadoop"]} }
{ "update": { "_id": "2"} }
{ "doc" : {"tag" : ["java"]} }
{ "update": { "_id": "3"} }
{ "doc" : {"tag" : ["hadoop"]} }
{ "update": { "_id": "4"} }
{ "doc" : {"tag" : ["java", "elasticsearch"]} }

```

### 2、搜索articleID为KDKE-B-9947-#kL5或QQPX-R-3956-#aD8的帖子

```
GET /forum/article/_search 
{
  "query": {
    "constant_score": {
      "filter": {
        "terms": {
          "articleID": [
            "KDKE-B-9947-#kL5",
            "QQPX-R-3956-#aD8"
          ]
        }
      }
    }
  }
}
```

### 3、搜索tag中包含java的帖子

```
GET /forum/article/_search
{
    "query" : {
        "constant_score" : {
            "filter" : {
                "terms" : { 
                    "tag" : ["java"]
                }
            }
        }
    }
}

```

### 4、搜索结果仅仅搜索tag只包含java的帖子

添加tag_cnt 字段 表示doc 中 tag的数量

```
POST /forum/article/_bulk
{ "update": { "_id": "1"} }
{ "doc" : {"tag_cnt" : 2} }
{ "update": { "_id": "2"} }
{ "doc" : {"tag_cnt" : 1} }
{ "update": { "_id": "3"} }
{ "doc" : {"tag_cnt" : 1} }
{ "update": { "_id": "4"} }
{ "doc" : {"tag_cnt" : 2} }
```

```
GET /forum/article/_search
{
  "query": {
    "constant_score": {
      "filter": {
        "bool": {
          "must": [
            {
              "term": {
                "tag_cnt": 1
              }
            },
            {
              "terms": {
                "tag": ["java"]
              }
            }
          ]
        }
      }
    }
  }
}
```

## 5、知识点

- 1、terms多值搜索
- 2、优化terms多值搜索的结果
- 3、相当于SQL中的in语句