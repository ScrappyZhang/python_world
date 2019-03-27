# 2 MySQL 语句

**MySQL语句的关键词不区分大小写，语句以逗号为结束符，--为MySQL语句注释行。本节的所有命令操作均可以在终端命令实操后显示查看，建议初学者可以在图形化界面同步查看以便加深理解。笔者曾从事于传统IT行业，所以本章全程贯穿一个京东电子产品的数据库实例来进行演示各个语句以便我们更好的理解。**

## 2.1 MySQL基本操作

### 1. 数据库操作

主要包括数据库的查看、创建、使用、删除操作。

* **查看当前有哪些数据库       **

show databases;

```
mysql> show databases;

+--------------------+
| Database           |
+--------------------+
| information_schema |
| mysql              |
| performance_schema |
| sys                |
+--------------------+
可以看到，目前数据库里有默认的四个数据库，其中：mysql为本地服务器的配置、sys为系统配置信息。
```

* **创建数据库 **create database 数据库名 \[其他选项\];

```MySQL
create database jing_dong;


-- create database jing_dong charset=utf8;
-- charset是指数据库所用的编码集
```

* **使用数据库:**use 数据库名

```
use jing_dong;
```

```
mysql> use jing_dong;
Reading table information for completion of table and column names
You can turn off this feature to get a quicker startup with -A

Database changed

可以看到提示，数据库发生了改变。
```

* **查看当前使用的数据库**

```
select database();
```

```
mysql> select database();
+------------+
| database() |
+------------+
| jing_dong  |
+------------+

可以看到我们正在使用的数据库是jing_dong
```

* **删除数据库     **drop database 数据库名；

```
drop database jing_dong;
```

```
mysql> drop database jing_dong;
Query OK, 0 rows affected (0.04 sec)
通过显示所有数据库的命令会发现jing_dong数据库已被删除。
mysql> show databases;
+--------------------+
| Database           |
+--------------------+
| information_schema |
| mysql              |
| performance_schema |
| sys                |
+--------------------+
```

### 2. 数据表操作

主要包括表的创建、删除、属性查看、显示当前数据库的表及属性、表头的增删改查。

* **创建表**

```
create table 表名称(列声明);
详解：
CREATE TABLE table_name(
    column1 datatype contrai,
    column2 datatype,
    column3 datatype,
    .....
    columnN datatype,
    PRIMARY KEY(one or more columns)
);
```

示例：创建商品种类表goods\_cates

```
create table goods_cates(
    id int unsigned primary key auto_increment not null,
    name varchar(40) not null
);
```

列id，代表商品种类id，取值类型为int unsigned，primary key指id列为主键， 值为自增auto\_increment ,且不能为空\(not null\)；  
列name，代表商品种类名称，取值为最长40的字符，不能为空  
**注意：最后一列的后面不能有逗号**

```
系统会提示：
Query OK, 0 rows affected (0.04 sec)
说明添加成功
```

* **显示表**

```
show tables;
```

可以看到goods\_cates表已在数据库中。

```
mysql> show tables;
+----------------+
| Tables_in_jing |
+----------------+
| goods_cates    |
+----------------+
1 row in set (0.00 sec)
```

* **查看表结构**

```
desc 表名;

例：
desc goods_cates;
```

可以看到goods\_cates表的表头信息：共有两个字段id和name，以及它们的详细信息。

```
mysql> desc goods_cates;
+-------+------------------+------+-----+---------+----------------+
| Field | Type             | Null | Key | Default | Extra          |
+-------+------------------+------+-----+---------+----------------+
| id    | int(10) unsigned | NO   | PRI | NULL    | auto_increment |
| name  | varchar(40)      | NO   |     | NULL    |                |
+-------+------------------+------+-----+---------+----------------+
2 rows in set (0.01 sec)
```

* **添加字段\(列\)**

```
alter table 表名 add 列名 类型;

例：假设刚才我们所统计的商品种类名字太长，需要简称，那么向商品种类表中增加列abbreviation（种类简称）
alter table goods_cates add abbreviation varchar(5);
```

```
终端返回以下信息，说明添加成功。
mysql> alter table goods_cates add abbreviation varchar(5);
Query OK, 0 rows affected (0.05 sec)
Records: 0  Duplicates: 0  Warnings: 0
```

通过查看表结构来看是否添加成功abbreviation。里面已经有了该列。

```
mysql> desc goods_cates;
+--------------+------------------+------+-----+---------+----------------+
| Field        | Type             | Null | Key | Default | Extra          |
+--------------+------------------+------+-----+---------+----------------+
| id           | int(10) unsigned | NO   | PRI | NULL    | auto_increment |
| name         | varchar(40)      | NO   |     | NULL    |                |
| abbreviation | varchar(5)       | YES  |     | NULL    |                |
+--------------+------------------+------+-----+---------+----------------+
3 rows in set (0.00 sec)
```

* **修改字段\(列\)**

```
不重命名版：alter table 表名 modify 列名 类型及约束;   

例：假设我们现在认为简写还是太长了，仅仅需要3个长度的字符就可以。
alter table goods_cates modify abbreviation varchar(3);
```

运行后，通过查看表结构，可以看到这一列的数据类型修改为varchar\(3\)了。

```
+--------------+------------------+------+-----+---------+----------------+
| Field        | Type             | Null | Key | Default | Extra          |
+--------------+------------------+------+-----+---------+----------------+
| id           | int(10) unsigned | NO   | PRI | NULL    | auto_increment |
| name         | varchar(40)      | NO   |     | NULL    |                |
| abbreviation | varchar(3)       | YES  |     | NULL    |                |
+--------------+------------------+------+-----+---------+----------------+
```

```
重命名版：alter table 表名 change 列原名 列新名 类型及约束;   

例：假设我们觉得表头里“abbreviation”这个单词太长了，想修改为“简称”。
alter table goods_cates change abbreviation 简称 varchar(3);
```

运行后，通过查看表结构，可以看到这一列的表头更改为“简称”了。

```
+--------+------------------+------+-----+---------+----------------+
| Field  | Type             | Null | Key | Default | Extra          |
+--------+------------------+------+-----+---------+----------------+
| id     | int(10) unsigned | NO   | PRI | NULL    | auto_increment |
| name   | varchar(40)      | NO   |     | NULL    |                |
| 简称    | varchar(3)       | YES  |     | NULL    |                |
+--------+------------------+------+-----+---------+----------------+
```

* **删除字段\(列\)**

```
alter table 表名 drop 列名;

例： 突然项目通知，我们的客户所需要的商品的名称一般都很容易记着，根本不需要简称，我们需要删除它。
alter table goods_cates drop 简称;
```

可以看到，已经删除：

```
+-------+------------------+------+-----+---------+----------------+
| Field | Type             | Null | Key | Default | Extra          |
+-------+------------------+------+-----+---------+----------------+
| id    | int(10) unsigned | NO   | PRI | NULL    | auto_increment |
| name  | varchar(40)      | NO   |     | NULL    |                |
+-------+------------------+------+-----+---------+----------------+
```

* **查看创建表的语句**

```
show create table 表名;

例：假如我们刚进入一个新环境拿到一个新账号需要操作，一般我们可以先看一下表的一些信息。
show create table goods_cates;
```

可以看到goods\_cates表的数据引擎为InnoDB，字符集为latin1 ；主键时字头id等等。

注：**一般建议在创建数据库时指定字符集类型，比如utf8 。 create database jing\_dong charset=utf8;**

![](/assets/database_charset.png)

* **删除表**

```
drop table 表名;

例：非常抱歉，项目通知设计中取消掉这一表。
drop table goods_cates;
```

通过查看表语句`show tables;`发现jing\_dong数据库没有这个表了，由于我们只创建了这一个表，所以提示为空。

```
mysql> show tables;
Empty set (0.00 sec)
```

**好了，我们可以重新删除掉jing\_dong数据库，从一开始就定义数据库字符集为utf8来不断演练以上两节的操作了。**

### 3. 数据增删改查

以上两节相信我们已经可以在数据库里拥有表了。本小节主要来说明在表内增加数据、删除数据、修改数据和简单查询数据。

本小节讲解时会用到以下数据，先列出。大家也可以去之前提到的源码网址获取。

```
id,name
1,台式机
2,平板电脑
3,服务器/工作站
4,游戏本
5,笔记本
6,笔记本配件
7,超级本
8,路由器
9,交换机
10,网卡
```

* **添加数据行**

```
insert [into] 表名 [(列名1, 列名2, 列名3, ...)] values (值1, 值2, 值3, ...);

例：
insert into goods_cates values (0, '台式机');
```

注：也可以一次insert命令添加多条数据行：`insert into goods_cates values (0, '台式机'),(0, '平板电脑');`

* **查询数据行**

```
select 列名称 from 表名称 [查询条件];

例：查询我们添加的所有数据
select * from goods_cates;
```

注： \* 号代表全部.

可以查询看到我们添加的“台式机”。其中，id在台式机那一行为1，是因为我们之前在建立表格时设置其为自增的int数据，因此系统会默认为其赋值，我们一般用0进行占位。

```
+----+-----------+
| id | name      |
+----+-----------+
|  1 | 台式机    |
+----+-----------+
```

**拓展：**我们可以重复利用insert语句将以上提到的1条语句全部添加到数据库，但是很麻烦；所以我们把所有数据存入文件并**通过文件导入。**

```
load data local infile '/path/goods_cates.txt' into table goods_cates;

其中：path为文件存在路径。
```

通过查询所有来查看数据，已经成功导入。

```
mysql> select * from goods_cates;
+----+---------------------+
| id | name                |
+----+---------------------+
|  1 | 台式机               |
|  2 | 平板电脑             |
|  3 | 服务器/工作站         |
|  4 | 游戏本               |
|  5 | 笔记本               |
|  6 | 笔记本配件            |
|  7 | 超级本               |
|  8 | 路由器               |
|  9 | 交换机               |
| 10 | 网卡                 |
+----+---------------------+
```

* **修改数据行**

```
update 表名称 set 列名称=新值 where 更新条件;

例：
update goods_cates set name='柔性手机' where id=10;
```

系统返回信息一行已更新，如下：

```
Query OK, 1 row affected (0.00 sec)
Rows matched: 1  Changed: 1  Warnings: 0
```

通过查询操作，会发现id=10的数据行中name列的值已修改。

```
mysql> select * from goods_cates;
+----+---------------------+
| id | name                |
+----+---------------------+
|  1 | 台式机              |
|  2 | 平板电脑            |
|  3 | 服务器/工作站       |
|  4 | 游戏本              |
|  5 | 笔记本              |
|  6 | 笔记本配件          |
|  7 | 超级本              |
|  8 | 路由器              |
|  9 | 交换机              |
| 10 | 柔性手机            |
+----+---------------------+
```

* **删除数据行**

```
delete from 表名 where 条件

例：
delete from goods_cates where id=10;
```

系统会告诉我们，有一行数据有形象，如

```
Query OK, 1 row affected (0.00 sec)
```

通过查询语句来查看，我们发现柔性手机那一行已被删除。

```
mysql> select * from goods_cates;
+----+---------------------+
| id | name                |
+----+---------------------+
|  1 | 台式机              |
|  2 | 平板电脑            |
|  3 | 服务器/工作站       |
|  4 | 游戏本              |
|  5 | 笔记本              |
|  6 | 笔记本配件          |
|  7 | 超级本              |
|  8 | 路由器              |
|  9 | 交换机              |
+----+---------------------+
```

### 4. 数据备份与恢复命令

和平常生活一样，我们都喜欢备份和恢复，所以在基本操作最后一小节里列出备份和恢复命令。

* **备份**

运行mysqldump命令

```
示例：
mysqldump –uroot –p 数据库名 > python.sql;

# 按提示输入mysql的密码
```

* **恢复**

连接mysql，创建新的数据库

退出连接，执行如下命令：

```
mysql -uroot –p 新数据库名 < python.sql

# 根据提示输入mysql密码
```

## 2.2 MySQL查询

上一节中演示了基本的select \* from goods\_cates查询，可以获取到goods\_\_cates里的所有数据。但是当数据量大时，就需要各种合适的查询方式来获取指定的信息并呈现给我们。本节就来讲述并演示这些操作。在开始之间我们先介绍几个会用到的命令。

* **查询指定列\(指定字段\)**

```
select 列1,列2,... from 表名;
例:
select name from goods_cates;
```

```
+---------------------+
| name                |
+---------------------+
| 台式机              |
| 平板电脑            |
| 服务器/工作站       |
| 游戏本              |
| 笔记本              |
| 笔记本配件          |
| 超级本              |
| 路由器              |
| 交换机              |
| 网卡                |
+---------------------+
```

* **查询指定列，并在显示中为它指定别名**

```
select id as 序号, name as 名字 from goods_cates;
```

```
+--------+---------------------+
| 序号   | 名字                |
+--------+---------------------+
|      1 | 台式机              |
|      2 | 平板电脑            |
|      3 | 服务器/工作站       |
|      4 | 游戏本              |
|      5 | 笔记本              |
|      6 | 笔记本配件          |
|      7 | 超级本              |
|      8 | 路由器              |
|      9 | 交换机              |
|     10 | 网卡                |
+--------+---------------------+
```

注： as也可以为表等起别名以便简化语句等等。

* **查询时，显示时消除重复列**

在select后面列前使用distinct可以消除重复的行

```
select distinct 列1,... from 表名;
```

我们在goods\_cates表中先添加一列id=15，name=网卡的数据。

```
insert into goods_cates values (15, '网卡');



+----+---------------------+
| id | name                |
+----+---------------------+
|  1 | 台式机              |
|  2 | 平板电脑            |
|  3 | 服务器/工作站       |
|  4 | 游戏本              |
|  5 | 笔记本              |
|  6 | 笔记本配件          |
|  7 | 超级本              |
|  8 | 路由器              |
|  9 | 交换机              |
| 10 | 网卡                |
| 15 | 网卡                |
+----+---------------------+
```

接下来，我们获取name列，但不显示id。

```
mysql> select name from goods_cates;
+---------------------+
| name                |
+---------------------+
| 台式机              |
| 平板电脑            |
| 服务器/工作站       |
| 游戏本              |
| 笔记本              |
| 笔记本配件          |
| 超级本              |
| 路由器              |
| 交换机              |
| 网卡                |
| 网卡                |
+---------------------+
```

可以看到，显示了两行网卡。我们通过distinct来消除重复并显示，就剩一个网卡了。哇，针实用，就像EXCEL里面去重选项那样来识别本列有多少种选项。

```
mysql> select distinct name from goods_cates;
+---------------------+
| name                |
+---------------------+
| 台式机              |
| 平板电脑            |
| 服务器/工作站       |
| 游戏本              |
| 笔记本              |
| 笔记本配件          |
| 超级本              |
| 路由器              |
| 交换机              |
| 网卡                |
+---------------------+
```

### 1. 条件查询

细心的朋友会发现我们上一节删除数据行时已经实用过where关键词，没错，它就是用来条件查询的一个重要关键词。一般使用where子句来筛选获取其后语句为True的数据行。其查询的语句格式为：

```
select * from 表名 where 条件;
例：
select * from grands_goods where id=1;
```

where后面支持多种运算符，进行条件的处理

* 比较运算符
* 逻辑运算符
* 模糊查询
* 范围查询
* 空判断

接下来我们一一演示说明。

* **比较运算符**

```
等于: =
大于: >
大于等于: >=
小于: <
小于等于: <=
不等于: !=
```

示例：查询id大于7的商品种类

```
select * from goods_cates where id > 7;
```

```
+----+-----------+
| id | name      |
+----+-----------+
|  8 | 路由器    |
|  9 | 交换机    |
| 10 | 网卡      |
| 15 | 网卡      |
+----+-----------+
```

* **逻辑运算符**

```
and   与
or    或
not   非
建议为了程序逻辑易读性，可以结合括号操作
```

示例：查询name=网卡，并且id=10的商品种类

```
select * from goods_cates where name="网卡" and id = 10;
```

```
+----+--------+
| id | name   |
+----+--------+
| 10 | 网卡   |
+----+--------+
```

* **模糊查询**

```
关键字： like

% 表示任意多个任意字符
_ 表示一个任意字符
```

示例：查询商品种类name中包含“本”的数据

```
select * from goods_cates where name like '%本';
```

```
+----+-----------+
| id | name      |
+----+-----------+
|  4 | 游戏本    |
|  5 | 笔记本    |
|  7 | 超级本    |
+----+-----------+
```

* **范围查询**

```
in表示在一个非连续的范围内

between ... and ...表示在一个连续的范围内
```

示例：查询商品种类id为1、3、5的数据

```
select * from goods_cates where id in(1,3,5);
```

```
+----+---------------------+
| id | name                |
+----+---------------------+
|  1 | 台式机              |
|  3 | 服务器/工作站       |
|  5 | 笔记本              |
+----+---------------------+
```

示例：查询商品种类id为1到5的数据

```
select * from goods_cates where id between 1 and 5;
```

```
+----+---------------------+
| id | name                |
+----+---------------------+
|  1 | 台式机              |
|  2 | 平板电脑            |
|  3 | 服务器/工作站       |
|  4 | 游戏本              |
|  5 | 笔记本              |
+----+---------------------+
```

* **空判断**

```
判空is null
注意：null与''是不同的
```

为了方便后续操作演示，除了上一节的表之外，我们新增商品表goods,并向其插入数据。

```
create table goods(
    id int unsigned primary key auto_increment not null,
    name varchar(40) default '',
    price decimal(5,2),
    cate_id int unsigned,
    brand_id int unsigned,
    is_show bit default 1,
    is_saleoff bit default 0
);
```

```
insert into goods values(0,'r510vc 15.6英寸笔记本','笔记本','华硕','3399',default,default); 
insert into goods values(0,'y400n 14.0英寸笔记本电脑','笔记本','联想','4999',default,default);
insert into goods values(0,'g150th 15.6英寸游戏本','游戏本','雷神','8499',default,default); 
insert into goods values(0,'x550cc 15.6英寸笔记本','笔记本','华硕','2799',default,default); 
insert into goods values(0,'x240 超极本','超级本','联想','4880',default,default); 
insert into goods values(0,'u330p 13.3英寸超极本','超级本','联想','4299',default,default); 
insert into goods values(0,'svp13226scb 触控超极本','超级本','索尼','7999',default,default); 
insert into goods values(0,'ipad mini 7.9英寸平板电脑','平板电脑','苹果','1998',default,default);
insert into goods values(0,'ipad air 9.7英寸平板电脑','平板电脑','苹果','3388',default,default); 
insert into goods values(0,'ipad mini 配备 retina 显示屏','平板电脑','苹果','2788',default,default); 
insert into goods values(0,'ideacentre c340 20英寸一体电脑 ','台式机','联想','3499',default,default); 
insert into goods values(0,'vostro 3800-r1206 台式电脑','台式机','戴尔','2899',default,default); 
insert into goods values(0,'imac me086ch/a 21.5英寸一体电脑','台式机','苹果','9188',default,default); 
insert into goods values(0,'at7-7414lp 台式电脑 linux ）','台式机','宏碁','3699',default,default); 
insert into goods values(0,'z220sff f4f06pa工作站','服务器/工作站','惠普','4288',default,default); 
insert into goods values(0,'poweredge ii服务器','服务器/工作站','戴尔','5388',default,default); 
insert into goods values(0,'mac pro专业级台式电脑','服务器/工作站','苹果','28888',default,default); 
insert into goods values(0,'hmz-t3w 头戴显示设备','笔记本配件','索尼','6999',default,default); 
insert into goods values(0,'商务双肩背包','笔记本配件','索尼','99',default,default); 
insert into goods values(0,'x3250 m4机架式服务器','服务器/工作站','ibm','6888',default,default); 
insert into goods values(0,'商务双肩背包','笔记本配件','索尼','99',default,default);
```

商品表的信息为：

![](/assets/goods_information.png)

### 2. 排序查询

生活中经常会遇到将商品的价格进行排序，数据也一样。在行业分析数据库中，经常要将竞争对手的产品统计起来并和自己的产品一起排序。

MySQL中使用order by来进行排序查询。

```
select * from 表名 order by 列1 asc|desc [,列2 asc|desc,...]
```

##### 说明 {#说明}

* 将行数据按照列1进行排序，如果某些行列1的值相同时，则按照列2排序，以此类推
* 默认按照列值从小到大排列（asc）
* asc从小到大排列，即升序
* desc从大到小排序，即降序

示例： 将所有在库商品安装价格从小到大来排序显示

```
select * from goods order by price;
```

![](/assets/order_by.png)

### 3. 集合\(统计\)函数

几乎所有的数据处理软件都会有简单的数据统计功能，mysql也不例外。本小节介绍总数、最大、最小、总和和平均值五个简单的统计函数。

* **总数**

count\(\*\)表示计算总行数，括号中写星与列名，结果是相同的

示例：显示仓库当前在库商品数量

```
select count(*) from goods;
```

![](/assets/mysql_count.png)

* **最大**

max\(列\)表示求此列的最大值

示例：显示笔记本类商品中最大的id号

```
select max(id) from goods where cate_name="笔记本";
```

![](/assets/mysql_max.png)

* **最小**

min\(列\)表示求此列的最小值

示例：显示台式机类商品中最小的id号

```
select min(id) from goods where cate_name="台式机";
```

![](/assets/mysql_min.png)

* **总和**

sum\(列\)表示求此列的和

示例：显示目前在库台式机类商品总价

```
select sum(price) from goods where cate_name="台式机";
```

![](/assets/mysql_sum.png)

* **平均值**

avg\(列\)表示求此列的平均值

示例1：显示目前在库台式机类商品平均价格

```
select avg(price) from goods where cate_name="台式机";
```

![](/assets/mysql_avg.png)

示例2：显示目前在库台式机类商品平均价格，格式为小数点后两位

```
select round(avg(price), 2) as "平均价" from goods where cate_name="台式机";
```

![](/assets/mysql_xiaoshu.png)

### 4. 分组

生活中经常需要分组统计众多的商品名目，MySQL采用group by来进行分组查询。

* **group by**

将查询结果按照指定的字段\(列\)进行分组，内容相同的为一组

示例：按照产品种类来查看目前的商品都有哪些种类的产品

`select cate_name from goods group by cate_name;`

![](/assets/mysql_groupby1.png)

* **group by + group\_concat\(name\)**

将分组后的name字段信息按照分组结果打印出来

示例：按照产品种类来分组显示商品名称

`select cate_name, group_concat(name) from goods group by cate_name;`

![](/assets/mysql_groupby2.png)

* **group by + 集合函数**

示例：按照产品种类来分组，并显示各自的在库商品数量

`select cate_name, count(*) from goods group by cate_name;`

![](/assets/mysql_groupby3.png)

* **group by + having**

分组之后的条件查询

示例：按照产品按照产品种类来分组，并显示各自的在库商品数量大于3的商品id，

`select cate_name, group_concat(id) from goods group by cate_name having count(*) > 3;`

![](/assets/mysql_groupby4.png)

**having与where类似，可以筛选数据，where后的表达式怎么写，having后就怎么写  
where针对表中的列发挥作用，查询数据  
having对查询结果中的列发挥作用，筛选数据  
select id,name,price as s from goods having s&gt;2000 ;  
这里不能用where因为s是查询结果，而where只能对表中的字段名筛选**

* **group by + with rollup**

在最后新增一行，来记录当前列记录的总和

示例：按照产品按照产品种类来分组，并显示各自的在库商品数量大于3的商品id，且在最后一行列出所有商品id

`select cate_name, group_concat(id) from goods group by cate_name with rollup having count(*) > 3;`

![](/assets/mysql_groupby5.png)

### 5.部分行查询

![](/assets/mysql_limit.png)

如上图，我们的goods表仅仅21行数据却已经不能全部显示了。因此我们常常需要去显示部分行数据。MySQL里用limit来实现此功能。

`elect * from 表名 limit start,count;`

说明：从start开始，获取count条数据

示例：显示第1行到第10行的数据

`select * from goods limit 1, 10;`

![](/assets/mysql_limit1.png)

### 6. 连接查询

本小节数据已之前采用goods表和下面那个只有一行数据的goods\_cates表来进行演示说明。

![](/assets/mysql_join.png)

* **左连接  left join**

以左表为准，去右表找数据，如果没有匹配的数据，则以null补空位，所以输出结果数&gt;=左表原数据数

示例：

`select * from goods left join goods_cates on goods.cate_name= goods_cates.name;`

![](/assets/mysql_leftjoin.png)

* **右连接 right join**

  a left join b 等价于 b right join a    推荐使用左连接代替右连接。

* **内连接 inner join**

查询结果是左右连接的交集，即左右连接的结果去除null项后的集。

示例：

`select * from goods inner join goods_cates on goods.cate_name= goods_cates.name;`

![](/assets/mysql_innerjoin.png)

### 7. 自关联查询

**自连接查询其实等同于连接查询，需要两张表，只不过它的左表（父表）和右表（子表）都是自己。请记住这个。**

在之前的例子里我们使用了goods表和goods\_cates表来表明仓库中的商品信息和商品种类。假如我们现在需要来做一个产品层级分类：一级分类为商品种类，二级分类为商品名；商品信息仅仅展示id和名字。那么按照之前那样，我们建立两张表goods\_1和goods\_cates\_1，每张表表头均是id、 name、 cate\_name 。由于商品种类为一级分类，所以我们记其cate\_name=null。

```
create table goods_1(  
    id smallint primary key auto_increment,  
    name varchar(20) not null,  
    cate_name varchar(20)
);  

create table goods_cates_1(  
    id smallint primary key auto_increment,  
    name varchar(20) not null,  
    cate_name varchar(20) 
);  

insert into goods_1 values(1,'r510vc 15.6英寸笔记本','笔记本');
insert into goods_1 values(0,'y400n 14.0英寸笔记本电脑','笔记本');
insert into goods_1 values(0,'x550cc 15.6英寸笔记本','笔记本');
insert into goods_1 values(0,'x240 超极本','超级本');
insert into goods_1 values(0,'u330p 13.3英寸超极本','超级本');
insert into goods_1 values(0,'vp13226scb 触控超极本','超级本');

insert into goods_cates_1 values(1, '笔记本', null);
insert into goods_cates_1 values(0, '超极本', null);
```

两张表信息如下：

![](/assets/mysql_joinself.png)

goods\_1表仅能看到产品的上一级分类，并不能看见最终分类，最终分类信息在goods\_cates\_1表里。现在我们需要显示每一个产品完整的两层层级信息，怎么做呢？用上一节的连接命令将两张表信息连接起来就可以啦：

```
select goods_cates_1.cate_name,goods_1.cate_name,goods_1.name
from goods_1 left join goods_cates_1 on goods_1.cate_name = goods_cates_1.name;
```

![](/assets/mysql_joinself2.png)

假如产品分类层级有10层，我们又需要建立剩下的八张表。通过观察发现，这两张表的结构基本相同，那么以上功能能否用一张表实现呢？答案是肯定的。

我们现在来创建一张表goods\_cates\_2用于产品层级信息存储，它的字段有：编号——id、名字——name、所属种类id——cate\_id。其中默认顶级分类的所属种类为null。

```
create table goods_cates_2(  
    id smallint primary key auto_increment,  
    name varchar(20) not null,  
    cate_id smallint
); 

insert into goods_cates_2 values(1, '笔记本', null);
insert into goods_cates_2 values(2, '超极本', null);
insert into goods_cates_2 values(0,'r510vc 15.6英寸笔记本',1);
insert into goods_cates_2 values(0,'y400n 14.0英寸笔记本电脑',1);
insert into goods_cates_2 values(0,'x550cc 15.6英寸笔记本',1);
insert into goods_cates_2 values(0,'x240 超极本',2);
insert into goods_cates_2 values(0,'u330p 13.3英寸超极本',2);
insert into goods_cates_2 values(0,'vp13226scb 触控超极本',2);
```

![](/assets/mysql_joinself4.png)

现在我们假设有两张表gc2为goods\_cates\_2代表第二级分类的信息，等价于上面两张表例子的goods\_1表，gc1为goods\_cates\_2代表第一级分类的信息，等价于上面两张表例子中的goods\_cates\_1表。使用上面例子中的方法，通过左连接来显示完整的两层层级信息。如下图所示，语句中的元素实现了一一对应。也就是说我们给goods\_cates\_2表起了两个别名gc1和gc2，让它们来连接实现完整的两层层级信息。功能和我们上面两张表的方案一样，且如果有10层分类，我们并不需要新建表。![](/assets/mysql_joinself5.png)

这和本节开始时说的完全一样，对，这就是自连接。如下为操作命令示例。其中gc1 和gc2 为我们为自连接表启的别名，语法中要求select的显示列需要用别名来区分。

```
select gc1.cate_id,gc1.name,gc2.name 
from goods_cates_2 gc2 left join goods_cates_2 gc1 on gc2.cate_id=gc1.id;
```

![](/assets/mysql_joinself6.png)

### 8. 子查询

之前我们所看到的所有查询语句均是单一命令的原始查询，并没有基于某一个查询结果的二次查询语句。所以这一小节我们引入子查询这一概念，来做基于某一个查询结果的二次查询操作。

我们知道查询基本语句为：`select A from B where C;`

对应的子查询语句分别为

* **where型子查询**

**把内层查询结果当作外层查询的比较条件**

`select A from B where (查询语句);`

示例：显示商品表内的一组商品名称和商品种类数据，这些数据满足以下条件：商品种类的id号在商品种类表中小于2

`select name, cate_name from goods where cate_name in (select name from goods_cates where id < 3);`

![](/assets/mysql_wheresele.png)

* **from型子查询**

**把内层的查询结果供外层再次查询**

`select A from (查询语句) as D where C;`

示例： 显示一组数据中商品种类为笔记本的数据，这批数据来自于id小于5的商品表数据集合。

`select name, cate_name from (select * from goods where id < 5) as t where t.cate_name = "笔记本";`

![](/assets/mysqlfrom1.png)

* **exists型子查询**

**把外层查询获得的行数据逐条传递到内层查询中被调用的地方，若内层查询非空则保留该条行数据，若内层查询为空则舍弃该条行数据。**

`select A from B where exists(查询语句)；`

示例： 从goods\_cates表中显示种类姓名，但是这些种类姓名必须满足一个条件，就是在goods表中存在此种类的商品。

`select name from goods_cates where exists(select * from goods where goods.cate_name = goods_cates.name);`

![](/assets/mysqlexists.png)

### 9. 小结

查询的综合完整语句：



```
  SELECT select_expr [,select_expr,...] [
  FROM tb\_name

  \[WHERE 条件判断\]

  \[GROUP BY {col\_name \| postion} \[ASC \| DESC\], ...\] 

  \[HAVING WHERE 条件判断\]

  \[ORDER BY {col\_name\|expr\|postion} \[ASC \| DESC\], ...\]

  \[ LIMIT {\[offset,\]rowcount \| row\_count OFFSET offset}\]
  ]
```



