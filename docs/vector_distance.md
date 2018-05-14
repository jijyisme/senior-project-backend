# Vector Distance

**URL** : `/vector_distance/`

**Method** : `POST`

**Auth required** : No

**Permissions required** : None

## Success Response

**Code** : `200 OK`

**Content examples**

```json
{
  "string_list": ["ฉัน", "กิน", "ข้าว"],
  "vector_list": ["[1, 0, 2, ...]", "[5, 1, 2, ...]", "[4, 6, 1, ...]"],
  "distances": [{"w1": "ฉัน", "w2": "กิน", "distance": 0.7}, {"w1": "ฉัน", "w2": "กิน", "distance": 0.7}]
}
```

