import discord
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import os

# MODELİ YÜKLE
model = load_model("keras_Model.h5", compile=False)
class_names = [line.strip() for line in open("labels.txt", "r").readlines()]

# TAHMİN FONKSİYONU
def predict_image(image_path):
    size = (224, 224)
    image = Image.open(image_path).convert("RGB")
    image = ImageOps.fit(image, size, Image.Resampling.LANCOS)

    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    prediction = model.predict(data)[0]
    predictions = [(class_names[i], float(prediction[i])) for i in range(len(class_names))]
    predictions.sort(key=lambda x: x[1], reverse=True)
    
    return predictions

# YORUM METNİ OLUŞTUR
def get_comment(top_class, confidence):
    if confidence > 0.85:
        if "taze" in top_class.lower():
            return f"🍏 Bu meyve oldukça **taze** görünüyor, rahatlıkla yiyebilirsin! (%{confidence*100:.1f}) ✅"
        else:
            return f"🍂 Bu meyve büyük ihtimalle **çürümüş**... dikkatli ol. (%{confidence*100:.1f}) ❌"
    elif confidence > 0.6:
        return f"🤔 Bu meyve **{top_class}** olabilir ama çok da emin değilim. Yemeni önermem. (%{confidence*100:.1f})"
    else:
        return f"🤷 Emin değilim, o yüzden yemenizi önermem. Belki **{top_class}** olabilir. (%{confidence*100:.1f})"

# BOT AYARLARI
intents = discord.Intents.default()
intents.message_content = True
intents.messages = True
intents.guilds = True

bot = discord.Client(intents=intents)

# MESAJ ALGILAYICI
@bot.event
async def on_message(message):
    if message.author == bot.user:
        return

    if message.attachments:
        for attachment in message.attachments:
            if attachment.filename.lower().endswith((".png", ".jpg", ".jpeg")):
                image_path = "temp.jpg"
                await attachment.save(image_path)

                try:
                    predictions = predict_image(image_path)
                    
                    # Tüm sınıfları yüzde olarak sırala
                    lines = [f"🔬 **Model Tahmini:**"]
                    for cls, score in predictions:
                        lines.append(f"- **{cls}**: %{score*100:.1f}")

                    # En yüksek olasılığa göre yorum ekle
                    top_class, top_score = predictions[0]
                    comment = get_comment(top_class, top_score)

                    await message.channel.send("\n".join(lines) + "\n\n" + comment)

                except Exception as e:
                    await message.channel.send(f"❌ Tahmin hatası: {e}")
                finally:
                    os.remove(image_path)

# BOT TOKENİNİ BURAYA YAZ
bot.run("BOT_TOKENİNİ_BURAYA_YAZ")
