import telebot
from io import BytesIO
import requests
from PIL import Image
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch

# Загрузка предварительно обученной модели
model = torch.load("./model.pb")

# Инициализация бота
with open("t.txt") as f:
    token = f.readline().strip()
bot = telebot.TeleBot(token)

# Список пород
classes = [
    "affenpinscher",
    "afghan hound",
    "african hunting dog",
    "airedale",
    "american staffordshire terrier",
    "appenzeller",
    "australian terrier",
    "basenji",
    "basset",
    "beagle",
    "bedlington terrier",
    "bernese mountain dog",
    "black-and-tan coonhound",
    "blenheim spaniel",
    "bloodhound",
    "bluetick",
    "border collie",
    "border terrier",
    "borzoi",
    "boston bull",
    "bouvier des flandres",
    "boxer",
    "brabancon griffon",
    "briard",
    "brittany spaniel",
    "bull mastiff",
    "cairn",
    "cardigan",
    "chesapeake bay retriever",
    "chihuahua",
    "chow",
    "clumber",
    "cocker spaniel",
    "collie",
    "curly-coated retriever",
    "dandie dinmont",
    "dhole",
    "dingo",
    "doberman",
    "english foxhound",
    "english setter",
    "english springer",
    "entlebucher",
    "eskimo dog",
    "flat-coated retriever",
    "french bulldog",
    "german shepherd",
    "german short-haired pointer",
    "giant schnauzer",
    "golden retriever",
    "gordon setter",
    "great dane",
    "great pyrenees",
    "greater swiss mountain dog",
    "groenendael",
    "ibizan hound",
    "irish setter",
    "irish terrier",
    "irish water spaniel",
    "irish wolfhound",
    "italian greyhound",
    "japanese spaniel",
    "keeshond",
    "kelpie",
    "kerry blue terrier",
    "komondor",
    "kuvasz",
    "labrador retriever",
    "lakeland terrier",
    "leonberg",
    "lhasa",
    "malamute",
    "malinois",
    "maltese dog",
    "mexican hairless",
    "miniature pinscher",
    "miniature poodle",
    "miniature schnauzer",
    "newfoundland",
    "norfolk terrier",
    "norwegian elkhound",
    "norwich terrier",
    "old english sheepdog",
    "otterhound",
    "papillon",
    "pekinese",
    "pembroke",
    "pomeranian",
    "pug",
    "redbone",
    "rhodesian ridgeback",
    "rottweiler",
    "saint bernard",
    "saluki",
    "samoyed",
    "schipperke",
    "scotch terrier",
    "scottish deerhound",
    "sealyham terrier",
    "shetland sheepdog",
    "shih-tzu",
    "siberian husky",
    "silky terrier",
    "soft-coated wheaten terrier",
    "staffordshire bullterrier",
    "standard poodle",
    "standard schnauzer",
    "sussex spaniel",
    "tibetan mastiff",
    "tibetan terrier",
    "toy poodle",
    "toy terrier",
    "vizsla",
    "walker hound",
    "weimaraner",
    "welsh springer spaniel",
    "west highland white terrier",
    "whippet",
    "wire-haired fox terrier",
    "yorkshire terrier"
]

# Обработка изображения
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Функция для предсказания породы
def classify_image(image):

    # Получение изображения из сообщения пользователя
    image = Image.open(BytesIO(image))
    image = transform(image).unsqueeze(0)

    # Обработка изображения и предсказание
    output = model(image)
    probs = torch.softmax(model(image), dim=1)
    conf = torch.max(probs[0]) * 100
    top_prob, top_class = probs.topk(1)
    return probs, conf

# Получение сообщений и выполнение предсказания
@bot.message_handler(content_types=["photo"])
def handle_message(message):
    # загрузка изображения из сообщения
    file_info = bot.get_file(message.photo[-1].file_id)
    image_url = 'https://api.telegram.org/file/bot{0}/{1}'.format(bot.token, file_info.file_path)
    image = requests.get(image_url).content

    # классификация изображения
    prob, conf = classify_image(image)

    # Отправка ответа пользователю
    breed = classes[torch.argmax(prob).item()]
    if breed[0] in ["a", "e", "i", "o", "u"]:
        article = "an"
    else:
        article = "a"
    if conf <= 50:
        answer = f"It’s tricky! I'm not sure, but maybe it is {article} {breed}!"
    elif 51 <= conf <= 79:
        answer = f"I think it is {article} {breed}!"
    else:
        answer = f"I'm pretty sure it is {article} {breed}!"
    bot.reply_to(message, answer)

@bot.message_handler(commands=["help"])
def help_function(message):
    """Отправляем список доступных команд."""
    help_message = "I know 120 dog breeds and I can try to guess the breed in the photo! 🐕  \nSend me a picture of a dog so I can start guessing!\nList of bot commands:\n/start - get started\n/help - output this hint"
    bot.send_message(message.chat.id, help_message)

@bot.message_handler(commands=["start"])
def start_function(message):
    """Отправляем список доступных команд."""
    start_message = "I know 120 dog breeds and I can try to guess the breed in the photo! 🐕 \nSend me a picture of a dog so I can start guessing!\nList of bot commands:\n/start - get started\n/help - output this hint"
    bot.send_message(message.chat.id, start_message)

@bot.message_handler(content_types=["text"])
def error_message(message):
    """Отправляем сообщение об ошибке."""
    error_message = "Unfortunately, I can't understand you 😟\nI can identify dog breeds in the pictures you send, but I don't talk. \nPlease send me a picture or ask for help with the command /help"
    bot.send_message(message.chat.id, error_message)

@bot.message_handler(content_types=["document"])
def error_message(message):
    """Отправляем сообщение об ошибке."""
    error_message = "Please send a picture as a photo, not as a document."
    bot.send_message(message.chat.id, error_message)

# запуск бота
bot.polling()
