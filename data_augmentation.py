import os
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img # type: ignore
import logging
import shutil

# Configurar logger
logging.basicConfig(level=logging.INFO)

# Diretórios de entrada e saída
input_dir = 'teste' # Diretório com as imagens originais
gray_dir = 'data/gray' # Diretório para armazenar as imagens em escala de cinza
blur_dir = 'data/blur' # Diretório para armazenar as imagens com filtro de desfoque
edge_dir = 'data/edge' # Diretório para armazenar as imagens com bordas detectadas
all_images_dir = 'data/all_images' # Diretório para armazenar todas as imagens
augmented_dir = 'data/augmented' # Diretório para armazenar as imagens aumentadas

# Número de imagens aumentadas por imagem original
num_augmented_images = 10  # Ajuste conforme necessário

# Criação dos diretórios
os.makedirs(gray_dir, exist_ok=True)
os.makedirs(blur_dir, exist_ok=True)
os.makedirs(edge_dir, exist_ok=True)
os.makedirs(all_images_dir, exist_ok=True)
os.makedirs(augmented_dir, exist_ok=True)

# Parâmetros de data augmentation
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    brightness_range=[0.8, 1.2],
    channel_shift_range=20.0
)

# Função para aplicar escala de cinza e filtros
def apply_filters(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    kernel_size = (5, 5)
    blurred_image = cv2.GaussianBlur(image, kernel_size, 0)
    edged_image = cv2.Canny(gray_image, 30, 150)
    return gray_image, blurred_image, edged_image

# Função para salvar as imagens filtradas
def save_filtered_images(input_dir, gray_dir, blur_dir, edge_dir):
    for filename in os.listdir(input_dir):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            img_path = os.path.join(input_dir, filename)
            img = cv2.imread(img_path)
            if img is None:
                logging.warning(f"Failed to read {img_path}. Skipping.")
                continue
            
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Aplica os filtros
            gray_img, blurred_img, edged_img = apply_filters(img_rgb)
            
             # Salvando as imagens
            gray_img_filename = os.path.join(gray_dir, f'{os.path.splitext(filename)[0]}_gray.jpg')
            blurred_img_filename = os.path.join(blur_dir, f'{os.path.splitext(filename)[0]}_blurred.jpg')
            edged_img_filename = os.path.join(edge_dir, f'{os.path.splitext(filename)[0]}_edged.jpg')

            cv2.imwrite(gray_img_filename, gray_img)
            cv2.imwrite(blurred_img_filename, cv2.cvtColor(blurred_img, cv2.COLOR_RGB2BGR))  # Blur no espaço de cores original
            cv2.imwrite(edged_img_filename, edged_img)

# Copiar todas as imagens originais para a pasta all_images
def copy_original_images_to_all_images(original_dir, all_images_dir):
    for filename in os.listdir(original_dir):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            shutil.copy(os.path.join(original_dir, filename), os.path.join(all_images_dir, filename))

# Copiar imagens das pastas gray e blur para a pasta all_images
def copy_filtered_images_to_all_images(gray_dir, blur_dir, all_images_dir):
    # Copiar imagens gray
    for filename in os.listdir(gray_dir):
        shutil.copy(os.path.join(gray_dir, filename), os.path.join(all_images_dir, filename))

    # Copiar imagens blur
    for filename in os.listdir(blur_dir):
        shutil.copy(os.path.join(blur_dir, filename), os.path.join(all_images_dir, filename))

# Função para aplicar data augmentation nas imagens
def apply_data_augmentation(all_images_dir, augmented_dir, datagen, num_augmented_images):
    for filename in os.listdir(all_images_dir):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            img_path = os.path.join(all_images_dir, filename)
            img = load_img(img_path)
            x = img_to_array(img)  # Converte a imagem para um array
            x = x.reshape((1,) + x.shape)           

            # Aplica data augmentation
            i = 0
            for batch in datagen.flow(x, batch_size=1, save_to_dir=augmented_dir,
                                      save_prefix=os.path.splitext(filename)[0], save_format='jpg'):
                i += 1
                if i >= num_augmented_images:
                    break  # Para após gerar a quantidade desejada de imagens por imagem original

# Processar as imagens
save_filtered_images(input_dir, gray_dir, blur_dir, edge_dir)

# Copiar imagens originais para o diretório all_images
copy_original_images_to_all_images(input_dir, all_images_dir)

# Copiar imagens gray e blur para o diretório all_images
copy_filtered_images_to_all_images(gray_dir, blur_dir, all_images_dir)

# Aplicar data augmentation na pasta all_images
apply_data_augmentation(all_images_dir, augmented_dir, datagen, num_augmented_images)