# Categorization

**URL** : `/categorization/`

**Method** : `POST`

**Auth required** : No

**Permissions required** : None

## Success Response

**Code** : `200 OK`

**Content examples**

```json
{
    "confidence_tag_list": [
      { "tag": "AIS", "confidence": 0.6 },
      { "tag": "โทรศัพท์มือถือ", "confidence": 0.5 }
    ],
}
```

## Notes

* If the text from input URL cannot be retrieved, return status 400 (BAD_REQUEST)
* If the text from input string is too large (>10000), return status 413 (REQUEST_ENTITY_TOO_LARGE)
