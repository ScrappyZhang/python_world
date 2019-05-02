# 11_ElasticSearch multi_match+most fiels策略进行multi-field搜索

## 概述

- 从best-fields换成most-fields策略
- **best-fields**策略，主要是说将**某一个field**匹配尽**可能多的关键词**的doc优先返回回来
- **most-fields**策略，主要是说尽可能返回**更多field**匹配到**某个关键词**的doc，优先返回回来

## 与best_fields的区别：

### best_fields

是对多个field进行搜索，挑选某个field匹配度最高的那个分数，同时在多个query最高分相同的情况下，在一定程度上考虑其他query的分数。简单来说，你对多个field进行搜索，就想搜索到某一个field尽可能包含更多关键字的数据

- 优点：通过best_fields策略，以及综合考虑其他field，还有minimum_should_match支持，可以尽可能精准地将匹配的结果推送到最前面
- 缺点：除了那些精准匹配的结果，其他差不多大的结果，排序结果不是太均匀，没有什么区分度了
- 实际的例子：百度之类的搜索引擎，最匹配的到最前面，但是其他的就没什么区分度

#### most_fields

综合多个field一起进行搜索，尽可能多地让所有field的query参与到总分数的计算中来，此时就会是个大杂烩，出现类似best_fields案例最开始的那个结果，结果不一定精准，某一个document的一个field包含更多的关键字，但是因为其他document有更多field匹配到了，所以排在了前面；所以需要建立类似title.std这样的field，尽可能让某一个field精准匹配query string，贡献更高的分数，将更精准匹配的数据排到前面

- 优点：将尽可能匹配更多field的结果推送到最前面，整个排序结果是比较均匀的
- 缺点：可能那些精准匹配的结果，无法推送到最前面
- 实际的例子：wiki，明显的most_fields策略，搜索结果比较均匀，但是的确要翻好几页才能找到最匹配的结果