# POC-coqui-tts-models-infer-with-SpeechT5
This repo mainly POC for infer with SpeechT5 vocoder in the coquiTTS codebase.

For the speechT5 infer codebase change in the: `.TTS/utils/synthesizer.py` script. 

In the `tts` method, a significant experiment was conducted using SpeechT5. This involved processing the `mel_postnet_spec` with a SpeechT5-based vocoder, `SpeechT5HifiGan`, to produce the final waveform. The following code snippet illustrates this experimental integration:

```python
def tts(
        self,
        text: str = "",
        speaker_name: str = "",
        language_name: str = "",
        speaker_wav=None,
        style_wav=None,
        style_text=None,
        reference_wav=None,
        reference_speaker_name=None,
    ) -> List[int]:
        """🐸 TTS magic. Run all the models and generate speech.

        Args:
            text (str): input text.
            speaker_name (str, optional): spekaer id for multi-speaker models. Defaults to "".
            language_name (str, optional): language id for multi-language models. Defaults to "".
            speaker_wav (Union[str, List[str]], optional): path to the speaker wav. Defaults to None.
            style_wav ([type], optional): style waveform for GST. Defaults to None.
            style_text ([type], optional): transcription of style_wav for Capacitron. Defaults to None.
            reference_wav ([type], optional): reference waveform for voice conversion. Defaults to None.
            reference_speaker_name ([type], optional): spekaer id of reference waveform. Defaults to None.
        Returns:
            List[int]: [description]
        """
        start_time = time.time()
        wavs = []

        if not text and not reference_wav:
            raise ValueError(
                "You need to define either `text` (for sythesis) or a `reference_wav` (for voice conversion) to use the Coqui TTS API."
            )

        if text:
            sens = self.split_into_sentences(text)
            print(" > Text splitted to sentences.")
            print(sens)

        # handle multi-speaker
        speaker_embedding = None
        speaker_id = None
        if self.tts_speakers_file or hasattr(self.tts_model.speaker_manager, "name_to_id"):

            # handle Neon models with single speaker.
            if len(self.tts_model.speaker_manager.name_to_id) == 1:
                speaker_id = list(self.tts_model.speaker_manager.name_to_id.values())[0]

            elif speaker_name and isinstance(speaker_name, str):
                if self.tts_config.use_d_vector_file:
                    # get the average speaker embedding from the saved d_vectors.
                    speaker_embedding = self.tts_model.speaker_manager.get_mean_embedding(
                        speaker_name, num_samples=None, randomize=False
                    )
                    speaker_embedding = np.array(speaker_embedding)[None, :]  # [1 x embedding_dim]
                else:
                    # get speaker idx from the speaker name
                    speaker_id = self.tts_model.speaker_manager.name_to_id[speaker_name]

            elif not speaker_name and not speaker_wav:
                raise ValueError(
                    " [!] Look like you use a multi-speaker model. "
                    "You need to define either a `speaker_name` or a `speaker_wav` to use a multi-speaker model."
                )
            else:
                speaker_embedding = None
        else:
            if speaker_name:
                raise ValueError(
                    f" [!] Missing speakers.json file path for selecting speaker {speaker_name}."
                    "Define path for speaker.json if it is a multi-speaker model or remove defined speaker idx. "
                )

        # handle multi-lingaul
        language_id = None
        if self.tts_languages_file or (
            hasattr(self.tts_model, "language_manager") and self.tts_model.language_manager is not None
        ):

            if len(self.tts_model.language_manager.name_to_id) == 1:
                language_id = list(self.tts_model.language_manager.name_to_id.values())[0]

            elif language_name and isinstance(language_name, str):
                language_id = self.tts_model.language_manager.name_to_id[language_name]

            elif not language_name:
                raise ValueError(
                    " [!] Look like you use a multi-lingual model. "
                    "You need to define either a `language_name` or a `style_wav` to use a multi-lingual model."
                )

            else:
                raise ValueError(
                    f" [!] Missing language_ids.json file path for selecting language {language_name}."
                    "Define path for language_ids.json if it is a multi-lingual model or remove defined language idx. "
                )

        # compute a new d_vector from the given clip.
        if speaker_wav is not None:
            speaker_embedding = self.tts_model.speaker_manager.compute_embedding_from_clip(speaker_wav)

        use_gl = self.vocoder_model is None

        if not reference_wav:
            for sen in sens:

                # print(" sen ", sen)
                # print(" style_wav ",style_wav)
                # print(" self.tts_config ",self.tts_config)
                # print(" self.use_cuda ", self.use_cuda )
                # print(" speaker_id ", speaker_id)
                # print(" style_wav ",style_wav)
                # print(" style_text ", style_text)
                # print(" use_gl ", use_gl)
                # print(" speaker_embedding ",speaker_embedding)
                # print(" language_id ",language_id)

                # synthesize voice
                outputs = synthesis(
                    model=self.tts_model,
                    text=sen,
                    CONFIG=self.tts_config,
                    use_cuda=self.use_cuda,
                    speaker_id=speaker_id,
                    style_wav=style_wav,
                    style_text=style_text,
                    use_griffin_lim=use_gl,
                    d_vector=speaker_embedding,
                    language_id=language_id,
                )
                waveform = outputs["wav"]
                # print(" waveform ", waveform)
                # print(" ---------------------- ")
                mel_postnet_spec = outputs["outputs"]["model_outputs"][0].detach().cpu().numpy()


                # print(" mel_postnet_spec shape ",mel_postnet_spec.shape)
                if not use_gl:
                    # """
                    # Default Start ->
                    # denormalize tts output based on tts audio config
                    mel_postnet_spec = self.tts_model.ap.denormalize(mel_postnet_spec.T).T
                    device_type = "cuda" if self.use_cuda else "cpu"
                    # renormalize spectrogram based on vocoder config
                    vocoder_input = self.vocoder_ap.normalize(mel_postnet_spec.T)
                    # compute scale factor for possible sample rate mismatch
                    scale_factor = [
                        1,
                        self.vocoder_config["audio"]["sample_rate"] / self.tts_model.ap.sample_rate,
                    ]
                    if scale_factor[1] != 1:
                        print(" > interpolating tts model output.")
                        vocoder_input = interpolate_vocoder_input(scale_factor, vocoder_input)
                    else:
                        vocoder_input = torch.tensor(vocoder_input).unsqueeze(0)  # pylint: disable=not-callable
                    # run vocoder model
                    # [1, T, C]
                    waveform = self.vocoder_model.inference(vocoder_input.to(device_type))

                    ## <- Default End
                    # """
                    
                    ''' ------------------- '''
                    """
                    # Experiment with SpeechT5 
                    
                    # mel_postnet_spec = self.tts_model.ap.denormalize(mel_postnet_spec.T).T
                    # device_type = "cuda" if self.use_cuda else "cpu"
                    # renormalize spectrogram based on vocoder config
                    # vocoder_input = self.vocoder_ap.normalize(mel_postnet_spec.T)
                    
                    # vocoder_input = vocoder(torch.from_numpy(self.vocoder_ap.normalize(mel_postnet_spec.T)).unsqueeze(0).transpose(1, 2))
                    # vocoder_input = vocoder(torch.from_numpy(mel_postnet_spec.T).unsqueeze(0).transpose(1, 2))
                    
                    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
                    vocoder_input = vocoder(torch.from_numpy(mel_postnet_spec))
                    print(vocoder_input)
                    print("ok -------")

                    import soundfile as sf
                    sf.write("/home/asif/tts_all/coqui_tts/aushik_bhai_backups/inference_audio_test/infer_w_speechT5.wav", vocoder_input.detach().numpy(), samplerate=48000, format='wav')

                    assert False
                    # compute scale factor for possible sample rate mismatch
                    # scale_factor = [
                    #     1,
                    #     self.vocoder_config["audio"]["sample_rate"] / self.tts_model.ap.sample_rate,
                    # ]
                    # if scale_factor[1] != 1:
                    #     print(" > interpolating tts model output.")
                    #     vocoder_input = interpolate_vocoder_input(scale_factor, vocoder_input)
                    # else:
                    #     vocoder_input = torch.tensor(vocoder_input).unsqueeze(0)  # pylint: disable=not-callable
                    # # run vocoder model
                    # # [1, T, C]
                    # waveform = self.vocoder_model.inference(vocoder_input.to(device_type))
                    
                    
                     # End Experiment with SpeechT5 
                    """

                if self.use_cuda and not use_gl:
                    waveform = waveform.cpu()
                if not use_gl:
                    waveform = waveform.numpy()
                waveform = waveform.squeeze()

                # trim silence
                if "do_trim_silence" in self.tts_config.audio and self.tts_config.audio["do_trim_silence"]:
                    waveform = trim_silence(waveform, self.tts_model.ap)

                wavs += list(waveform)
                wavs += [0] * 10000
        else:
            # get the speaker embedding or speaker id for the reference wav file
            reference_speaker_embedding = None
            reference_speaker_id = None
            if self.tts_speakers_file or hasattr(self.tts_model.speaker_manager, "name_to_id"):
                if reference_speaker_name and isinstance(reference_speaker_name, str):
                    if self.tts_config.use_d_vector_file:
                        # get the speaker embedding from the saved d_vectors.
                        reference_speaker_embedding = self.tts_model.speaker_manager.get_embeddings_by_name(
                            reference_speaker_name
                        )[0]
                        reference_speaker_embedding = np.array(reference_speaker_embedding)[
                            None, :
                        ]  # [1 x embedding_dim]
                    else:
                        # get speaker idx from the speaker name
                        reference_speaker_id = self.tts_model.speaker_manager.name_to_id[reference_speaker_name]
                else:
                    reference_speaker_embedding = self.tts_model.speaker_manager.compute_embedding_from_clip(
                        reference_wav
                    )
            outputs = transfer_voice(
                model=self.tts_model,
                CONFIG=self.tts_config,
                use_cuda=self.use_cuda,
                reference_wav=reference_wav,
                speaker_id=speaker_id,
                d_vector=speaker_embedding,
                use_griffin_lim=use_gl,
                reference_speaker_id=reference_speaker_id,
                reference_d_vector=reference_speaker_embedding,
            )
            waveform = outputs
            if not use_gl:
                mel_postnet_spec = outputs[0].detach().cpu().numpy()
                # denormalize tts output based on tts audio config
                mel_postnet_spec = self.tts_model.ap.denormalize(mel_postnet_spec.T).T
                device_type = "cuda" if self.use_cuda else "cpu"
                # renormalize spectrogram based on vocoder config
                vocoder_input = self.vocoder_ap.normalize(mel_postnet_spec.T)
                # compute scale factor for possible sample rate mismatch
                scale_factor = [
                    1,
                    self.vocoder_config["audio"]["sample_rate"] / self.tts_model.ap.sample_rate,
                ]
                if scale_factor[1] != 1:
                    print(" > interpolating tts model output.")
                    vocoder_input = interpolate_vocoder_input(scale_factor, vocoder_input)
                else:
                    vocoder_input = torch.tensor(vocoder_input).unsqueeze(0)  # pylint: disable=not-callable
                # run vocoder model
                # [1, T, C]
                waveform = self.vocoder_model.inference(vocoder_input.to(device_type))
            if self.use_cuda:
                waveform = waveform.cpu()
            if not use_gl:
                waveform = waveform.numpy()
            wavs = waveform.squeeze()

        # compute stats
        process_time = time.time() - start_time
        audio_time = len(wavs) / self.tts_config.audio["sample_rate"]
        print(f" > Processing time: {process_time}")
        print(f" > Real-time factor: {process_time / audio_time}")
        return wavs
```