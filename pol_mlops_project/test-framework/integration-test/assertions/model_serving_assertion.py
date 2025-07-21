from databricks.sdk.service.serving import QueryEndpointResponse


def assert_response(response:QueryEndpointResponse) :
    prediction = float(response.predictions[0])
    assert 8.0 <= prediction <= 9.0, f"Value out of range: {prediction}"
