# Keyword Expansion

**URL** : `/keyword_expansion/`

**Method** : `POST`

**Auth required** : No

**Permissions required** : None

## Success Response

**Code** : `200 OK`

**Content examples**

```json
{
  "string_list": ["ไอโฟน", "Iphone", "ไอโฟน5"],
  "similarity_list": [0.75, 0.70, 0.60],
}
```

## Notes

* If the text from input URL cannot be retrieved, return status 400 (BAD_REQUEST)
* If the text from input string is too large (>10000), return status 413 (REQUEST_ENTITY_TOO_LARGE)
