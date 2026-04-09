import requests
import os

url = 'http://127.0.0.1:5000/predict'
file_path = r'D:\SVTR-Project\Data\Raw\Kitkat.jpg'

def sending_test():
    if not os.path.exists(file_path):
        print(f"Error: File tidak ditemukan di {file_path}")
        return

    try:
        with open(file_path, 'rb') as f:
            files = {'image': f}
            print(f"Mengirim gambar ke {url}...")
            
            response = requests.post(url, files=files)
            
            if response.status_code == 200:
                print("Berhasil!")
                print("Respon Server:", response.json())
            else:
                print(f"Gagal! Kode Status: {response.status_code}")
                print("Pesan Error:", response.text)

    except requests.exceptions.ConnectionError:
        print("Koneksi Gagal! Pastikan app.py sudah dijalankan di terminal sebelah.")
    except Exception as e:
        print(f"Terdapat kesalahan: {e}")

if __name__ == "__main__":
    sending_test()