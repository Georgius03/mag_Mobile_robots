import socket, requests
import yaml


with open('parameters.yaml') as config_file:
    config = yaml.safe_load(config_file)

    
# Подключение к Robotino
def connect_to_robotino():
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((config['socket_params']['IP_ADDRESS'], config['socket_params']['PORT']))
        print("Successfully connected")
        return sock
    except Exception as e:
        print(f"Error connecting: {e}")
        return None

# Отправка управляющих сигналов -
def send_velocity(vx:float, vy:float, omega:float):
    url = f"http://{config['socket_params']['IP_ADDRESS']}/data/omnidrive"
    data = [vx, vy, omega]
    try:
        response = requests.post(url, json=data)
        if response.status_code == 200:
            # print(f"Sent Vx: {vx}, Vy: {vy}, w: {omega}")
            pass
        else:
            print(f"Send error: {response.status_code} - {response.text}")
            pass
    except Exception as e:
        print(f"Error sending data: {e}")