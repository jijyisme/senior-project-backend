from django.conf.urls import url, include
# from rest_framework import routers
from nlp_tools import views

# router = routers.DefaultRouter()
# router.register(r'tokenize', views.tokenize)

# Wire up our API using automatic URL routing.
# Additionally, we include login URLs for the browsable API.
urlpatterns = [
    # url(r'^', include(router.urls)),
    # url(r'^api-auth/', include('rest_framework.urls', namespace='rest_framework')),
    url(r'^tokenize', views.get_token),
    url(r'^vectorize', views.get_vector),
    url(r'^ner', views.get_ner)
]
