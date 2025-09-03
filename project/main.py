import kivy
kivy.require('1.9.1')

from kivy.app import App
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.textinput import TextInput
from kivy.uix.floatlayout import FloatLayout
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.uix.image import Image
from kivy.core.window import Window

import cv2
import numpy as np
from os import makedirs, listdir
from os.path import exists, join

# --- CONFIGURAÇÃO DA JANELA ---
Window.clearcolor = (0.2, 0.2, 0.2, 1)
Window.size = (980, 720)

# --- DIRETÓRIO DE FACES ---
data_path = 'faces/'
if not exists(data_path):
    makedirs(data_path)

# --- FUNÇÃO PARA DETECTAR FACES ---
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
def face_extractor(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 0:
        return None
    x, y, w, h = faces[0]
    return img[y:y+h, x:x+w]

# --- CLASSE PARA CÂMERA ---
class KivyCV(Image):
    def __init__(self, capture, fps, callback=None, **kwargs):
        super().__init__(**kwargs)
        self.capture = capture
        self.callback = callback
        Clock.schedule_interval(self.update, 1.0 / fps)

    def update(self, dt):
        ret, frame = self.capture.read()
        if ret:
            face = face_extractor(frame)
            if face is not None and self.callback:
                self.callback(face)
            buf = cv2.flip(frame, 0).tobytes()
            image_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            image_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            self.texture = image_texture

# --- LOGIN ---
class LoginScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        layout = FloatLayout()

        self.login_input = TextInput(hint_text="Login", size_hint=(0.6, 0.08), pos_hint={'x':0.2,'y':0.6})
        self.password_input = TextInput(hint_text="Senha", password=True, size_hint=(0.6,0.08), pos_hint={'x':0.2,'y':0.48})

        login_btn = Button(text="Confirmar identidade com foto", size_hint=(0.6,0.08), pos_hint={'x':0.2,'y':0.36})
        login_btn.bind(on_press=self.identificar)

        criar_btn = Button(text="Criar conta", size_hint=(0.28,0.08), pos_hint={'x':0.2,'y':0.24})
        criar_btn.bind(on_press=lambda x: setattr(self.manager, 'current', 'create_account'))

        esqueceu_btn = Button(text="Esqueceu a senha?", size_hint=(0.28,0.08), pos_hint={'x':0.52,'y':0.24})
        esqueceu_btn.bind(on_press=lambda x: setattr(self.manager, 'current', 'reset_request'))

        layout.add_widget(self.login_input)
        layout.add_widget(self.password_input)
        layout.add_widget(login_btn)
        layout.add_widget(criar_btn)
        layout.add_widget(esqueceu_btn)
        self.add_widget(layout)

    def identificar(self, instance):
        self.manager.get_screen('face_recognition').set_login(self.login_input.text)
        self.manager.current = 'face_recognition'

# --- CREATE ACCOUNT ---
class CreateAccountScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        layout = FloatLayout()

        self.email = TextInput(hint_text="Email", size_hint=(0.6,0.08), pos_hint={'x':0.2,'y':0.7})
        self.login = TextInput(hint_text="Login", size_hint=(0.6,0.08), pos_hint={'x':0.2,'y':0.58})
        self.password = TextInput(hint_text="Senha", password=True, size_hint=(0.6,0.08), pos_hint={'x':0.2,'y':0.46})
        self.face_taken = False
        self.capture_count = 0
        self.capturing = False

        capture_btn = Button(text="Cadastro com foto", size_hint=(0.6,0.08), pos_hint={'x':0.2,'y':0.34})
        capture_btn.bind(on_press=self.start_capture)

        criar_btn = Button(text="Criar conta", size_hint=(0.6,0.08), pos_hint={'x':0.2,'y':0.22})
        criar_btn.bind(on_press=self.criar_conta)

        layout.add_widget(self.email)
        layout.add_widget(self.login)
        layout.add_widget(self.password)
        layout.add_widget(capture_btn)
        layout.add_widget(criar_btn)
        self.add_widget(layout)

        # Preview da câmera
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.kivy_cam = KivyCV(capture=self.cap, fps=30, callback=self.capture_frame)
        self.kivy_cam.size_hint = (0.3,0.3)
        self.kivy_cam.pos_hint = {'x':0.65,'y':0.05}
        layout.add_widget(self.kivy_cam)

    def start_capture(self, instance):
        if not self.capturing:
            self.capturing = True
            self.capture_count = 0
            print("Iniciando captura de 100 fotos...")

    def capture_frame(self, face):
        if self.capturing and face is not None:
            if self.capture_count < 100:
                self.capture_count += 1
                cv2.imwrite(f"{data_path}{self.login.text}_{self.capture_count}.png",
                            cv2.resize(face,(200,200)))
                print(f"Foto {self.capture_count}/100 capturada")
            else:
                self.capturing = False
                self.face_taken = True
                print("Captura de 100 fotos concluída!")

    def criar_conta(self, instance):
        if self.email.text and self.login.text and self.password.text and self.face_taken:
            print("Conta criada com sucesso!")
            self.manager.current = "login"
        else:
            print("Preencha todos os campos e capture a foto!")

# --- RESET REQUEST ---
class ResetRequestScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        layout = FloatLayout()
        Label(text="Digite o e-mail para enviar a redefinição de senha", size_hint=(0.8,0.08), pos_hint={'x':0.1,'y':0.7}, font_size=20)
        self.email_input = TextInput(hint_text="Email", size_hint=(0.6,0.08), pos_hint={'x':0.2,'y':0.6})
        enviar_btn = Button(text="Enviar", size_hint=(0.6,0.08), pos_hint={'x':0.2,'y':0.48})
        enviar_btn.bind(on_press=lambda x: print(f"Enviar redefinição para {self.email_input.text}"))
        voltar_btn = Button(text="Voltar", size_hint=(0.2,0.08), pos_hint={'x':0.05,'y':0.9})
        voltar_btn.bind(on_press=lambda x: setattr(self.manager,'current','login'))

        layout.add_widget(self.email_input)
        layout.add_widget(enviar_btn)
        layout.add_widget(voltar_btn)
        self.add_widget(layout)

# --- RESET PASSWORD ---
class ResetPasswordScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        layout = FloatLayout()
        Label(text="Redefinir senha", size_hint=(0.8,0.08), pos_hint={'x':0.1,'y':0.7}, font_size=24)
        self.senha = TextInput(hint_text="Senha", password=True, size_hint=(0.6,0.08), pos_hint={'x':0.2,'y':0.6})
        self.confirma = TextInput(hint_text="Confirmar Senha", password=True, size_hint=(0.6,0.08), pos_hint={'x':0.2,'y':0.48})
        enviar_btn = Button(text="Enviar", size_hint=(0.6,0.08), pos_hint={'x':0.2,'y':0.36})
        enviar_btn.bind(on_press=self.redefinir)
        voltar_btn = Button(text="Voltar", size_hint=(0.2,0.08), pos_hint={'x':0.05,'y':0.9})
        voltar_btn.bind(on_press=lambda x: setattr(self.manager,'current','login'))

        layout.add_widget(self.senha)
        layout.add_widget(self.confirma)
        layout.add_widget(enviar_btn)
        layout.add_widget(voltar_btn)
        self.add_widget(layout)

    def redefinir(self, instance):
        if self.senha.text and self.confirma.text and self.senha.text == self.confirma.text:
            print("Senha redefinida!")
            self.manager.current = 'login'
        else:
            print("Preencha corretamente os campos!")

# --- FACE RECOGNITION ---
class FaceRecognitionScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        layout = FloatLayout()
        self.capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.kivy_cam = KivyCV(capture=self.capture, fps=30, callback=self.verificar_usuario)
        self.kivy_cam.size_hint = (0.8,0.8)
        self.kivy_cam.pos_hint = {'x':0.1,'y':0.1}
        layout.add_widget(self.kivy_cam)

        self.label = Label(text="Posicione seu rosto", size_hint=(0.8,0.08), pos_hint={'x':0.1,'y':0.05}, font_size=24)
        layout.add_widget(self.label)

        voltar_btn = Button(text="Voltar", size_hint=(0.2,0.08), pos_hint={'x':0.05,'y':0.9})
        voltar_btn.bind(on_press=lambda x: setattr(self.manager, 'current', 'login'))
        layout.add_widget(voltar_btn)

        self.add_widget(layout)
        self.login = None

    def set_login(self, login):
        self.login = login

    def verificar_usuario(self, face):
        if not self.login:
            self.label.text = "Informe o login primeiro!"
            return

        user_faces = [f for f in listdir(data_path) if f.startswith(self.login)]
        if not user_faces:
            self.label.text = "USUÁRIO NÃO CADASTRADO"
            return

        recognized = False
        face_gray = cv2.cvtColor(cv2.resize(face,(200,200)), cv2.COLOR_BGR2GRAY)
        for f in user_faces:
            img = cv2.imread(join(data_path,f), cv2.IMREAD_GRAYSCALE)
            model = cv2.face.LBPHFaceRecognizer_create()
            model.train([img], np.array([0]))
            result = model.predict(face_gray)
            confidence = 100 * (1 - result[1]/300)
            if confidence > 75:
                recognized = True
                break

        self.label.text = "IDENTIFICADO" if recognized else "NÃO IDENTIFICADO"

# --- SCREEN MANAGER ---
sm = ScreenManager()
sm.add_widget(LoginScreen(name="login"))
sm.add_widget(CreateAccountScreen(name="create_account"))
sm.add_widget(ResetRequestScreen(name="reset_request"))
sm.add_widget(ResetPasswordScreen(name="reset_password"))
sm.add_widget(FaceRecognitionScreen(name="face_recognition"))

# --- APP PRINCIPAL ---
class SistemaApp(App):
    def build(self):
        return sm

if __name__ == "__main__":
    SistemaApp().run()
