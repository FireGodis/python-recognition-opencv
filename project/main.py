import os
import cv2
import numpy as np
import kivy

kivy.require('1.11.1')

from kivy.app import App
from kivy.clock import Clock
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.graphics.texture import Texture

# --- Preparação ---
if not os.path.exists("faces"):
    os.makedirs("faces")

if not os.path.exists("cadastros"):
    os.makedirs("cadastros")

face_classifier = cv2.CascadeClassifier("lib/haarcascade_frontalface_default.xml")


# ------------ Tela de Login ------------
class LoginScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        layout = BoxLayout(orientation="vertical", padding=20, spacing=10)

        layout.add_widget(Label(text="Login", font_size=32))

        self.cpf_input = TextInput(hint_text="CPF", size_hint_y=None, height=40)
        layout.add_widget(self.cpf_input)

        self.senha_input = TextInput(hint_text="Senha", password=True, size_hint_y=None, height=40)
        layout.add_widget(self.senha_input)

        login_btn = Button(text="Login", size_hint_y=None, height=50)
        login_btn.bind(on_press=self.login_action)
        layout.add_widget(login_btn)

        self.status_label = Label(text="")
        layout.add_widget(self.status_label)

        botoes = BoxLayout(size_hint_y=None, height=50, spacing=10)
        criar_btn = Button(text="Criar conta")
        criar_btn.bind(on_press=lambda *_: setattr(self.manager, 'current', 'create_account'))
        esqueceu_btn = Button(text="Esqueceu a senha?")
        esqueceu_btn.bind(on_press=lambda *_: setattr(self.manager, 'reset_request'))

        botoes.add_widget(criar_btn)
        botoes.add_widget(esqueceu_btn)
        layout.add_widget(botoes)

        self.add_widget(layout)

    def login_action(self, instance):
        cpf = self.cpf_input.text.strip()
        senha = self.senha_input.text.strip()
        filepath = os.path.join("cadastros", f"{cpf}.txt")

        if not os.path.exists(filepath):
            self.status_label.text = "CPF não cadastrado."
            return

        with open(filepath, "r", encoding="utf-8") as f:
            dados = f.read().splitlines()
            dados_dict = {line.split(":")[0]: line.split(":")[1] for line in dados if ":" in line}

        if dados_dict.get("Senha") != senha:
            self.status_label.text = "Senha incorreta."
            return

        # Passou na senha → salvar CPF em uso e ir para reconhecimento
        self.manager.get_screen("recognition").cpf_logado = cpf
        self.manager.current = "recognition"


# ------------ Tela Criar Conta ------------
class CreateAccountScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        main_layout = BoxLayout(orientation="horizontal", padding=20, spacing=10)

        # Lado esquerdo (formulário)
        form_layout = GridLayout(cols=1, spacing=10, size_hint=(0.5, 1))

        form_layout.add_widget(Label(text="Criar Conta", font_size=28))

        self.nome_input = TextInput(hint_text="Nome")
        form_layout.add_widget(self.nome_input)

        self.cpf_input = TextInput(hint_text="CPF")
        form_layout.add_widget(self.cpf_input)

        self.cargo_input = TextInput(hint_text="Cargo")
        form_layout.add_widget(self.cargo_input)

        self.email_input = TextInput(hint_text="Email")
        form_layout.add_widget(self.email_input)

        self.senha_input = TextInput(hint_text="Senha", password=True)
        form_layout.add_widget(self.senha_input)

        self.progress_label = Label(text="Aguardando captura...")
        form_layout.add_widget(self.progress_label)

        capturar_btn = Button(text="Iniciar Captura de Rosto")
        capturar_btn.bind(on_press=self.start_capture)
        form_layout.add_widget(capturar_btn)

        salvar_btn = Button(text="Salvar Cadastro")
        salvar_btn.bind(on_press=self.save_account)
        form_layout.add_widget(salvar_btn)

        voltar_btn = Button(text="Voltar")
        voltar_btn.bind(on_press=lambda *_: setattr(self.manager, 'current', 'login'))
        form_layout.add_widget(voltar_btn)

        main_layout.add_widget(form_layout)

        # Lado direito (câmera)
        self.camera_widget = Image(size_hint=(0.5, 1))
        main_layout.add_widget(self.camera_widget)

        self.add_widget(main_layout)

        # Variáveis
        self.capture = None
        self.frames_captured = 0
        self.capturing = False

    def face_extractor(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)
        if len(faces) == 0:
            return None
        for (x, y, w, h) in faces:
            return img[y:y+h, x:x+w]

    def start_capture(self, instance):
        self.capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.capture.isOpened():
            self.progress_label.text = "Erro: não foi possível acessar a câmera."
            return

        self.frames_captured = 0
        self.capturing = True
        self.progress_label.text = "Analisando Rosto 0/100..."
        Clock.schedule_interval(self.update_camera, 1.0/30.0)

    def update_camera(self, dt):
        if self.capture and self.capturing:
            ret, frame = self.capture.read()
            if not ret:
                self.progress_label.text = "Erro ao capturar frame."
                self.stop_capture()
                return

            buf = cv2.flip(frame, 0).tobytes()
            texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            self.camera_widget.texture = texture

            face = self.face_extractor(frame)
            if face is not None:
                self.frames_captured += 1
                face = cv2.resize(face, (200, 200))
                face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

                cpf = self.cpf_input.text.strip()
                file_name_path = f"faces/{cpf}({self.frames_captured}).png"
                cv2.imwrite(file_name_path, face)

                if self.frames_captured < 100:
                    self.progress_label.text = f"Analisando Rosto {self.frames_captured}/100..."
                else:
                    self.progress_label.text = "Análise concluída! Fotos salvas."
                    self.stop_capture()

    def stop_capture(self):
        self.capturing = False
        if self.capture:
            self.capture.release()
            self.capture = None
        Clock.unschedule(self.update_camera)

    def save_account(self, instance):
        cpf = self.cpf_input.text.strip()
        filepath = os.path.join("cadastros", f"{cpf}.txt")

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"Nome:{self.nome_input.text}\n")
            f.write(f"CPF:{cpf}\n")
            f.write(f"Cargo:{self.cargo_input.text}\n")
            f.write(f"Email:{self.email_input.text}\n")
            f.write(f"Senha:{self.senha_input.text}\n")

        self.progress_label.text = "Cadastro salvo com sucesso!"


# ------------ Tela de Reconhecimento (Login com Face) ------------
class RecognitionScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cpf_logado = None

        layout = BoxLayout(orientation="vertical", padding=20, spacing=10)
        layout.add_widget(Label(text="Reconhecimento Facial", font_size=28))

        self.status_label = Label(text="Posicione seu rosto na câmera...")
        layout.add_widget(self.status_label)

        self.camera_widget = Image(size_hint=(1, 1))
        layout.add_widget(self.camera_widget)

        voltar_btn = Button(text="Voltar")
        voltar_btn.bind(on_press=lambda *_: setattr(self.manager, 'current', 'login'))
        layout.add_widget(voltar_btn)

        self.add_widget(layout)

        self.model = None
        self.capture = None
        self.event = None

    def on_enter(self, *args):
        cpf = self.cpf_logado
        if not cpf:
            self.status_label.text = "Erro: Nenhum CPF em uso."
            return

        self.capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.capture.isOpened():
            self.status_label.text = "Erro: não foi possível acessar a câmera"
            return

        self.train_model(cpf)
        self.event = Clock.schedule_interval(self.update_camera, 1.0 / 30.0)

    def on_leave(self, *args):
        if self.event:
            Clock.unschedule(self.event)
            self.event = None
        if self.capture:
            self.capture.release()
            self.capture = None

    def train_model(self, cpf):
        data_path = "faces/"
        onlyfiles = [f for f in os.listdir(data_path) if f.startswith(cpf)]

        Training_Data, Labels = [], []
        for i, file in enumerate(onlyfiles):
            image_path = os.path.join(data_path, file)
            images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            Training_Data.append(np.asarray(images, dtype=np.uint8))
            Labels.append(i)

        if len(Labels) == 0:
            self.status_label.text = "Nenhuma face cadastrada para este CPF."
            return

        Labels = np.asarray(Labels, dtype=np.int32)
        self.model = cv2.face.LBPHFaceRecognizer_create()
        self.model.train(np.asarray(Training_Data), np.asarray(Labels))

    def update_camera(self, dt):
        if not self.capture:
            return

        ret, frame = self.capture.read()
        if not ret:
            self.status_label.text = "Erro ao capturar frame"
            return

        buf = cv2.flip(frame, 0).tobytes()
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.camera_widget.texture = texture

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            roi = gray[y:y+h, x:x+w]
            roi = cv2.resize(roi, (200, 200))

            if self.model is not None:
                result = self.model.predict(roi)
                confidence = int(100 * (1 - (result[1]) / 300))

                if confidence > 75:
                    self.manager.current = "home"
                else:
                    self.status_label.text = f"Face não reconhecida ({confidence}%)"


# ------------ Tela Home ------------
class HomeScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        layout = BoxLayout(orientation="vertical", padding=20, spacing=10)
        self.label = Label(text="Login realizado com sucesso!", font_size=28)
        layout.add_widget(self.label)

        logout_btn = Button(text="Deslogar", size_hint=(None, None), size=(120, 40), pos_hint={"right": 1, "top": 1})
        logout_btn.bind(on_press=lambda *_: setattr(self.manager, 'current', 'login'))
        layout.add_widget(logout_btn)

        self.add_widget(layout)


# ------------ Tela Reset Senha ------------
class ResetRequestScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        layout = BoxLayout(orientation="vertical", padding=20, spacing=10)

        layout.add_widget(Label(text="Esqueceu a Senha?", font_size=28))

        self.email_input = TextInput(hint_text="Digite seu Email", size_hint_y=None, height=40)
        layout.add_widget(self.email_input)

        enviar_btn = Button(text="Enviar Email de Redefinição", size_hint_y=None, height=40)
        enviar_btn.bind(on_press=self.enviar_email)
        layout.add_widget(enviar_btn)

        self.confirm_label = Label(text="")
        layout.add_widget(self.confirm_label)

        voltar_btn = Button(text="Voltar", size_hint_y=None, height=40)
        voltar_btn.bind(on_press=lambda *_: setattr(self.manager, 'current', 'login'))
        layout.add_widget(voltar_btn)

        self.add_widget(layout)

    def enviar_email(self, instance):
        self.confirm_label.text = "Email de redefinição enviado!"


# ------------ App Principal ------------
class MyApp(App):
    def build(self):
        sm = ScreenManager()
        sm.add_widget(LoginScreen(name="login"))
        sm.add_widget(CreateAccountScreen(name="create_account"))
        sm.add_widget(RecognitionScreen(name="recognition"))
        sm.add_widget(HomeScreen(name="home"))
        sm.add_widget(ResetRequestScreen(name="reset_request"))
        return sm


if __name__ == "__main__":
    MyApp().run()
