"""Equal model names in different Django apps have distinct key spaces."""
from django.db import models
from fervis.host_api.adapters.django.schema_introspection import path_param_entity_target


def test_same_named_models_do_not_share_an_identity_namespace():
    def buyer_model(app_label):
        return type('Buyer', (models.Model,), {
            '__module__': __name__, 'buyer_id': models.UUIDField(primary_key=True),
            'Meta': type('Meta', (), {'app_label': app_label}),
        })
    global_buyer = buyer_model('global_accounts')
    store_buyer = buyer_model('store_accounts')
    global_target = path_param_entity_target(global_buyer, param_name='buyer_id')
    store_target = path_param_entity_target(store_buyer, param_name='buyer_id')
    assert global_target.entity_kind == 'global_accounts.buyer'
    assert store_target.entity_kind == 'store_accounts.buyer'
    assert global_target != store_target
