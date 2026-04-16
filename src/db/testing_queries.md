### 1. Посмотреть все типы узлов (labels)

```
CALL db.labels();
```

### 2. Посмотреть типы отношений

```
CALL db.relationshipTypes();
```

### 3. Посмотреть свойства узлов

```
MATCH (n:Entity)
RETURN n
LIMIT 10;
```

Проверка на наличие свойства `name`

### 4. Посмотреть свойства отношений

```
MATCH (a:Entity)-[r:RELATION]->(b:Entity)
RETURN a.name, r.type, b.name
LIMIT 25;
```

Проверка наличия `type` у отношения

### 5. Посмотреть структуру графа визуально

Самый удобный вариант:

```
MATCH (a)-[r]->(b)
RETURN a, r, b
LIMIT 50;
```

Neo4j отобразит граф.

### 6. Проверить количество узлов

```
MATCH (n:Entity)
RETURN count(n);
```

### 7. Проверить количество связей

```
MATCH ()-[r]->()
RETURN count(r);
```

### 8. Проверить конкретную сущность

```
MATCH (n:Entity {name: "NavTrust"})-[r]->(m)
RETURN n, r, m;
```

### 9. Посмотреть дубликаты сущностей (частая проблема)

```
MATCH (n:Entity)
WITH n.name AS name, count(*) AS cnt
WHERE cnt > 1
RETURN name, cnt;
```

Если что-то > 1 — значит MERGE работает не так, как ожидается.

### Быстрая sanity-проверка

Выполни последовательно:

```
CALL db.labels();
CALL db.relationshipTypes();
MATCH (n:Entity) RETURN count(n);
MATCH ()-[r]->() RETURN count(r);
MATCH (a)-[r]->(b) RETURN a,r,b LIMIT 25;
```

Если все выполняется без пустых результатов — загрузка прошла корректно.
