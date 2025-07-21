from databricks.sdk.service.serving import QueryEndpointResponse


def assert_response(response:QueryEndpointResponse) :
    assert len(response.predictions) == 2, response.predictions
