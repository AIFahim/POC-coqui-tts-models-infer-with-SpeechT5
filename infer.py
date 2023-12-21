import torchaudio
from TTS.utils.synthesizer import Synthesizer
import requests
import json
import base64
import wave
import numpy as np



# url = "https://dev.revesoft.com:6790/phonemizer"
# def infer(grapheme:str):
#     synth=Synthesizer(tts_checkpoint="/home/asif/coqui_allign_tts/TTS/tts_train_dir/run-January-23-2023_06+33AM-6e3f74fc/checkpoint_98000.pth",tts_config_path="/home/asif/coqui_allign_tts/TTS/tts_train_dir/run-January-23-2023_06+33AM-6e3f74fc/config.json", vocoder_checkpoint="/home/asif/coqui_allign_tts/TTS/vocoder_model/UNIVERSAL_V1/g_02500000",vocoder_config="/home/asif/coqui_allign_tts/TTS/vocoder_model/UNIVERSAL_V1/config.json")
#     # synth=Synthesizer(tts_checkpoint="/home/asif/coqui_allign_tts/TTS/tts_train_dir/run-January-23-2023_06+33AM-6e3f74fc/checkpoint_98000.pth",tts_config_path="/home/asif/coqui_allign_tts/TTS/tts_train_dir/run-January-23-2023_06+33AM-6e3f74fc/config.json")
#     payload = json.dumps({
#     "text": grapheme
#     })
#     headers = {
#     'Content-Type': 'application/json'
#     }

#     response = requests.request("POST", url, headers=headers, data=payload, verify=False)
#     phonme = response.json()['output']

#     phonme = " ".join(phonme)
#     # print("phonme ", phonme)
    
#     wav=synth.tts(phonme)
#     # synth.save_wav(wav,"/home/asif/testaudio/test_w_vocoder.wav")
#     # synth.save_wav(wav,"/home/asif/testaudio/test_w_o_vocoder.wav")

#     # with open("/home/asif/testaudio/test.wav", "rb") as f:
#     #     audio_data = f.read()

#     # base64_audio = base64.b64encode(audio_data).decode("utf-8") 
#     # return base64_audio 

    
# if __name__ == "__main__":
#     infer("গিনেস বুকে ১২ বার নাম লিখিয়েছেন মাগুরার মাহমুদুল")
#     # main("বাংলাদেশ","/home/asif/testaudio/")
    

def infer(grapheme:str, wav_file:str):
    
    # synth=Synthesizer(tts_checkpoint="/home/asif/tts_all/coqui_tts/aushik_bhai_backups/coqui_allign_tts_new_female_working/TTS/checkpoints_bn_female_new/run-June-15-2023_06+41AM-6e3f74fc/best_model_15387.pth",
    # tts_config_path="/home/asif/tts_all/coqui_tts/aushik_bhai_backups/coqui_allign_tts_new_female_working/TTS/checkpoints_bn_female_new/run-June-15-2023_06+41AM-6e3f74fc/config.json")

    synth=Synthesizer(tts_checkpoint="/home/asif/tts_all/coqui_tts/aushik_bhai_backups/coqui_allign_tts_new_female_working/TTS/checkpoints_bn_with_collect_male/run-October-29-2023_01+00PM-6e3f74fc/checkpoint_168000.pth",
    tts_config_path="/home/asif/tts_all/coqui_tts/aushik_bhai_backups/coqui_allign_tts_new_female_working/TTS/checkpoints_bn_with_collect_male/run-October-29-2023_01+00PM-6e3f74fc/config.json")

    # synth=Synthesizer(tts_checkpoint="/home/asif/tts_all/coqui_tts/aushik_bhai_backups/coqui_allign_tts_new_female_working/TTS/checkpoints_bn_female_new/run-June-15-2023_06+41AM-6e3f74fc/best_model_15387.pth",
    # tts_config_path="/home/asif/tts_all/coqui_tts/aushik_bhai_backups/coqui_allign_tts_new_female_working/TTS/checkpoints_bn_female_new/run-June-15-2023_06+41AM-6e3f74fc/config.json")
    
    wav=synth.tts(grapheme)
    synth.save_wav(wav, wav_file)
    

if __name__ == "__main__":
    #infer("আমি বাংলায় গান গাই।", "/home/elias/testaudio/test.wav")
    #infer("প্রতিটি ভোর মানেই নতুন এক নিদর্শন-পাখিদের কোলাহলে মুখরিত চারপাশ। ", "/home/elias/testaudio/test-03.wav")
    # infer("বড়গাঁওয়া মাঠে পরিমনির নৃত্য", "/home/asif/tts_all/coqui_tts/aushik_bhai_backups/inference_audio_test/test_w_vocoder_multibandmelgan_44k.wav")
    # infer("বড়গাঁওয়া মাঠে পরিমনির নৃত্য","/home/asif/tts_all/coqui_tts/aushik_bhai_backups/inference_audio_test/test_vits_44k_weight_123000.wav")

    infer("কেউ বলেন ঢাকা হলো জাদুর শহর, কেউ বলেন কবিতার শহর, কেউ বলেন কোলাহলের শহর, কেউ বলেন আলো আঁধারিতে হারিয়ে যাওয়ার শহর ইত্যাদি। আরও অনেক বিশেষণে বিশেষায়িত করা হয়েছে তাকে।","/home/asif/tts_all/TTS_Evaluations/TTS_Data_for_Comparisions/align_22k_collect_2nd_sample/inference_w_alignTTS_collect_male.wav")