from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello World from group 12"}


def test_predict():
    response = client.post("/predict/",
        json={"text": "https://answit.com/wp-content/uploads/2017/01/full-hd.jpg"}
    )
    json_data = response.json()

    assert response.status_code == 200
    assert json_data['generated_text'] == 'a lake with a waterfall and mountains '
