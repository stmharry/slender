import requests

data = {'images': ['http://member.healthyd.com/attachment/201111/12/2033_1321097217Zrg0.jpg']}
r = requests.post('http://dev.2bite.com:6006/classify/food_type', json=data)
print(r.text)
