import requests as r

url = "https://interpol.thebrainfox.com/api/random-person"
response = r.get(url)
if response.status_code == 200:
    person_data = response.json()
    print("Random Person Data from Interpol API:")
    print(person_data)
    print(f"image URL: https://interpol.thebrainfox.com/{person_data['photoUrl']}")

url = "https://interpol.thebrainfox.com/api/check-answer"

body = {
    "personId": person_data["id"],
    "userChoice": "linkedin"
}

response = r.post(url, json=body)  # <= ici
if response.status_code == 200:
    result_data = response.json()
    print("Check Answer Response:")
    print(result_data)
else:
    print(f"Error: {response.status_code} - {response.text}")
