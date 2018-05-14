# Word Embedding

**URL** : `/vectorize/`

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
}
```

## Notes

* If the text from input URL cannot be retrieved, return status 400 (BAD_REQUEST)
* If the text from input string is too large (>10000), return status 413 (REQUEST_ENTITY_TOO_LARGE)
