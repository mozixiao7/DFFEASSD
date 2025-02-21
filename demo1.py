#云雀恭弥
#开发时间:2025-01-15 21:23
import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert

def enve():
    # 加载两组音频文件
    audio_file_1 = 'D:\\研究生文献\\ASV\\LA\\ASVspoof2019_LA_train\\flac\\LA_T_1138215.flac'  # 第一组音频文件路径
    '''
    1000137
    1000618
    '''
    audio_file_2 = 'D:\\研究生文献\\ASV\\LA\\ASVspoof2019_LA_train\\flac\\LA_T_1000824.flac'  # 第二组音频文件路径

    y1, sr1 = librosa.load(audio_file_1, sr=16000)
    y2, sr2 = librosa.load(audio_file_2, sr=16000)

    # 计算包络：使用 Hilbert 变换
    def compute_envelope(signal):
        analytic_signal = hilbert(signal)
        return np.abs(analytic_signal)

    envelope1 = compute_envelope(y1)
    envelope2 = compute_envelope(y2)

    # 创建时域坐标
    time1 = np.arange(len(y1)) / sr1
    time2 = np.arange(len(y2)) / sr2



    # 绘制语音信号的波形图和包络图
    plt.figure(figsize=(9, 6))

    # 第一组语音信号波形图
    plt.subplot(2, 2, 1)
    plt.plot(time1, y1, label='Speech Signal 1', color='blue')
    plt.title('Speech Signal 1')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')

    # 第一组语音信号包络图
    plt.subplot(2, 2, 2)
    plt.plot(time1, envelope1, label='Envelope 1', color='red')
    plt.title('Envelope of Speech Signal 1')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')

    # 第二组语音信号波形图
    plt.subplot(2, 2, 3)
    plt.plot(time2, y2, label='Speech Signal 2', color='green')
    plt.title('Speech Signal 2')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')

    # 第二组语音信号包络图
    plt.subplot(2, 2, 4)
    plt.plot(time2, envelope2, label='Envelope 2', color='orange')
    plt.title('Envelope of Speech Signal 2')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')

    # 调整布局
    plt.tight_layout()
    plt.show()

def pitch():
    import librosa
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.interpolate import interp1d

    # 加载两组音频文件
    audio_file_1 = 'D:\\研究生文献\\ASV\\LA\\ASVspoof2019_LA_train\\flac\\LA_T_1138215.flac'  # 第一组音频文件路径
    '''
    1000137
    1000618
    '''
    audio_file_2 = 'D:\\研究生文献\\ASV\\LA\\ASVspoof2019_LA_train\\flac\\LA_T_1000824.flac'  # 第二组音频文件路径

    y1, sr1 = librosa.load(audio_file_1, sr=16000)
    y2, sr2 = librosa.load(audio_file_2, sr=16000)

    # 提取基频（Pitch）
    f0_1, _, _ = librosa.pyin(y1, fmin=librosa.note_to_hz('C1'), fmax=librosa.note_to_hz('C8'))
    f0_2, _, _ = librosa.pyin(y2, fmin=librosa.note_to_hz('C1'), fmax=librosa.note_to_hz('C8'))

    # 创建时域坐标
    time1 = np.arange(len(y1)) / sr1
    time2 = np.arange(len(y2)) / sr2

    # 对基频进行插值，使其与波形图的时间轴匹配
    interp_f0_1 = interp1d(np.linspace(0, len(f0_1) - 1, len(f0_1)), f0_1, kind='linear', fill_value="extrapolate")
    interp_f0_2 = interp1d(np.linspace(0, len(f0_2) - 1, len(f0_2)), f0_2, kind='linear', fill_value="extrapolate")

    # 插值后的基频数据
    f0_1_interpolated = interp_f0_1(np.arange(len(y1)))
    f0_2_interpolated = interp_f0_2(np.arange(len(y2)))

    # 绘制波形图和基频图
    plt.figure(figsize=(9, 6))

    # 第一组语音信号波形图
    plt.subplot(2, 2, 1)
    plt.plot(time1, y1, label='Speech Signal 1', color='blue')
    plt.title('Speech Signal 1 (Waveform)')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')

    # 第一组语音信号基频图
    plt.subplot(2, 2, 2)
    plt.plot(time1, f0_1_interpolated, label='F0 (Pitch) 1', color='green')
    plt.title('Fundamental Frequency (F0) of Speech Signal 1')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    # plt.ylim(0, 5000)  # 设置基频图的y轴范围，可以根据实际情况调整

    # 第二组语音信号波形图
    plt.subplot(2, 2, 3)
    plt.plot(time2, y2, label='Speech Signal 2', color='red')
    plt.title('Speech Signal 2 (Waveform)')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')

    # 第二组语音信号基频图
    plt.subplot(2, 2, 4)
    plt.plot(time2, f0_2_interpolated, label='F0 (Pitch) 2', color='orange')
    plt.title('Fundamental Frequency (F0) of Speech Signal 2')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    # plt.ylim(0, 5000)  # 设置基频图的y轴范围，可以根据实际情况调整

    # 调整布局
    plt.tight_layout()
    plt.show()

def zcr():
    import librosa
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.interpolate import interp1d

    # 加载两组音频文件
    audio_file_1 = 'D:\\研究生文献\\ASV\\LA\\ASVspoof2019_LA_train\\flac\\LA_T_1138215.flac'  # 第一组音频文件路径
    '''
    1000137
    1000618
    '''
    audio_file_2 = 'D:\\研究生文献\\ASV\\LA\\ASVspoof2019_LA_train\\flac\\LA_T_1000824.flac'  # 第二组音频文件路径

    y1, sr1 = librosa.load(audio_file_1, sr=16000)
    y2, sr2 = librosa.load(audio_file_2, sr=16000)

    # 提取零交叉率
    zcr_1 = librosa.feature.zero_crossing_rate(y1)[0]  # 提取第一组音频的零交叉率
    zcr_2 = librosa.feature.zero_crossing_rate(y2)[0]  # 提取第二组音频的零交叉率
    print(zcr_1,zcr_2)

    # 创建时域坐标
    time1 = np.arange(len(y1)) / sr1
    time2 = np.arange(len(y2)) / sr2

    # 对零交叉率进行插值，使其与波形图的时间轴匹配
    interp_zcr_1 = interp1d(np.linspace(0, len(zcr_1) - 1, len(zcr_1)), zcr_1, kind='linear', fill_value="extrapolate")
    interp_zcr_2 = interp1d(np.linspace(0, len(zcr_2) - 1, len(zcr_2)), zcr_2, kind='linear', fill_value="extrapolate")

    # 插值后的零交叉率数据
    zcr_1_interpolated = interp_zcr_1(np.arange(len(y1)))
    zcr_2_interpolated = interp_zcr_2(np.arange(len(y2)))

    # 绘制波形图和零交叉率图
    plt.figure(figsize=(9, 6))

    # 第一组语音信号波形图
    plt.subplot(2, 2, 1)
    plt.plot(time1, y1, label='Speech Signal 1', color='blue')
    plt.title('Speech Signal 1 (Waveform)')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')

    # 第一组语音信号零交叉率图
    plt.subplot(2, 2, 2)
    plt.plot(time1, zcr_1_interpolated, label='Zero Crossing Rate 1', color='green')
    plt.title('Zero Crossing Rate of Speech Signal 1')
    plt.xlabel('Time (s)')
    plt.ylabel('Zero Crossing Rate')
    # plt.ylim(0, 1)  # 零交叉率的范围通常在0到1之间

    # 第二组语音信号波形图
    plt.subplot(2, 2, 3)
    plt.plot(time2, y2, label='Speech Signal 2', color='red')
    plt.title('Speech Signal 2 (Waveform)')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')

    # 第二组语音信号零交叉率图
    plt.subplot(2, 2, 4)
    plt.plot(time2, zcr_2_interpolated, label='Zero Crossing Rate 2', color='orange')
    plt.title('Zero Crossing Rate of Speech Signal 2')
    plt.xlabel('Time (s)')
    plt.ylabel('Zero Crossing Rate')
    # plt.ylim(0, 1)  # 零交叉率的范围通常在0到1之间

    # 调整布局
    plt.tight_layout()
    plt.show()

def f0():
    import librosa
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.interpolate import interp1d

    # 加载两组音频文件
    audio_file_1 = 'D:\\研究生文献\\ASV\\LA\\ASVspoof2019_LA_train\\flac\\LA_T_1000406.flac'  # 第一组音频文件路径
    '''
    1000137
    1000618
    '''
    audio_file_2 = 'D:\\研究生文献\\ASV\\LA\\ASVspoof2019_LA_train\\flac\\LA_T_1001871.flac'  # 第二组音频文件路径

    y1, sr1 = librosa.load(audio_file_1, sr=16000)
    y2, sr2 = librosa.load(audio_file_2, sr=16000)

    # 提取基频（Pitch）使用 librosa.pyin
    f0_1, voiced_flag_1, voiced_probs_1 = librosa.pyin(
        y1,
        fmin=librosa.note_to_hz('C1'),
        fmax=librosa.note_to_hz('C8')
    )
    f0_2, voiced_flag_2, voiced_probs_2 = librosa.pyin(
        y2,
        fmin=librosa.note_to_hz('C1'),
        fmax=librosa.note_to_hz('C8')
    )

    # 创建时域坐标
    time1 = np.arange(len(y1)) / sr1
    time2 = np.arange(len(y2)) / sr2

    # 计算每个帧的时间位置
    hop_length = 512  # 默认的 hop_length 值
    frames1 = librosa.frames_to_time(np.arange(len(f0_1)), sr=sr1, hop_length=hop_length)
    frames2 = librosa.frames_to_time(np.arange(len(f0_2)), sr=sr2, hop_length=hop_length)

    # 对基频数据进行插值，使其与波形图的时间轴匹配
    def interpolate_f0(f0, original_time, target_time):
        # 移除未检测到的基频（NaN）
        valid_idx = ~np.isnan(f0)
        if np.sum(valid_idx) < 2:
            # 如果有效点不足，返回全零
            return np.zeros_like(target_time)
        interp_func = interp1d(
            original_time[valid_idx],
            f0[valid_idx],
            kind='linear',
            bounds_error=False,
            fill_value="extrapolate"
        )
        return interp_func(target_time)

    f0_1_interpolated = interpolate_f0(f0_1, frames1, time1)
    f0_2_interpolated = interpolate_f0(f0_2, frames2, time2)

    # 绘制波形图和基频轨迹图
    plt.figure(figsize=(9, 6))

    # 第一组语音信号波形图
    plt.subplot(2, 2, 1)
    plt.plot(time1, y1, label='Speech Signal 1', color='blue')
    plt.title('Speech Signal 1 (Waveform)')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()

    # 第一组语音信号基频轨迹图
    plt.subplot(2, 2, 2)
    plt.plot(time1, f0_1_interpolated, label='F0 (Pitch) 1', color='green')
    plt.title('Fundamental Frequency (F0) of Speech Signal 1')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    # plt.ylim(0, 500)  # 根据实际基频范围调整
    plt.legend()

    # 第二组语音信号波形图
    plt.subplot(2, 2, 3)
    plt.plot(time2, y2, label='Speech Signal 2', color='red')
    plt.title('Speech Signal 2 (Waveform)')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()

    # 第二组语音信号基频轨迹图
    plt.subplot(2, 2, 4)
    plt.plot(time2, f0_2_interpolated, label='F0 (Pitch) 2', color='orange')
    plt.title('Fundamental Frequency (F0) of Speech Signal 2')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    # plt.ylim(0, 500)  # 根据实际基频范围调整
    plt.legend()

    # 调整布局
    plt.tight_layout()
    plt.show()

def env_and_f0():
    import librosa
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.signal import hilbert
    from scipy.interpolate import interp1d

    # 加载两组音频文件
    audio_file_1 = 'D:\\研究生文献\\ASV\\LA\\ASVspoof2019_LA_train\\flac\\LA_T_1518499.flac'  # 第一组音频文件路径
    '''
    1000137
    1000618
    
    LA_T_8031274
    LA_T_9122421
    '''
    audio_file_2 = 'D:\\研究生文献\\ASV\\LA\\ASVspoof2019_LA_train\\flac\\LA_T_7228482.flac'  # 第二组音频文件路径

    y1, sr1 = librosa.load(audio_file_1, sr=16000)
    y2, sr2 = librosa.load(audio_file_2, sr=16000)

    # 提取基频（Pitch）使用 librosa.pyin
    f0_1, voiced_flag_1, voiced_probs_1 = librosa.pyin(
        y1,
        fmin=librosa.note_to_hz('C1'),
        fmax=librosa.note_to_hz('C8')
    )
    f0_2, voiced_flag_2, voiced_probs_2 = librosa.pyin(
        y2,
        fmin=librosa.note_to_hz('C1'),
        fmax=librosa.note_to_hz('C8')
    )

    # 计算包络：使用 Hilbert 变换
    def compute_envelope(signal):
        analytic_signal = hilbert(signal)
        return np.abs(analytic_signal)

    envelope1 = compute_envelope(y1)
    envelope2 = compute_envelope(y2)

    # 创建时域坐标
    time1 = np.arange(len(y1)) / sr1
    time2 = np.arange(len(y2)) / sr2

    # 计算每个帧的时间位置
    hop_length = 512  # 默认的 hop_length 值
    frames1 = librosa.frames_to_time(np.arange(len(f0_1)), sr=sr1, hop_length=hop_length)
    frames2 = librosa.frames_to_time(np.arange(len(f0_2)), sr=sr2, hop_length=hop_length)

    # 对基频数据进行插值，使其与波形图的时间轴匹配
    def interpolate_f0(f0, original_time, target_time):
        # 移除未检测到的基频（NaN）
        valid_idx = ~np.isnan(f0)
        if np.sum(valid_idx) < 2:
            # 如果有效点不足，返回全零
            return np.zeros_like(target_time)
        interp_func = interp1d(
            original_time[valid_idx],
            f0[valid_idx],
            kind='linear',
            bounds_error=False,
            fill_value="extrapolate"
        )
        return interp_func(target_time)

    f0_1_interpolated = interpolate_f0(f0_1, frames1, time1)
    f0_2_interpolated = interpolate_f0(f0_2, frames2, time2)

    # 绘制波形图、包络图和基频轨迹图
    plt.figure(figsize=(14, 12))

    # 第一组语音信号波形图
    plt.subplot(4, 2, 1)
    plt.plot(time1, y1, label='Speech Signal 1', color='blue')
    plt.title('Speech Signal 1 (Waveform)')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()

    # 第一组语音信号包络图
    plt.subplot(4, 2, 2)
    plt.plot(time1, envelope1, label='Envelope 1', color='red')
    plt.title('Envelope of Speech Signal 1')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()

    # 第一组语音信号基频轨迹图
    plt.subplot(4, 2, 3)
    plt.plot(time1, f0_1_interpolated, label='F0 (Pitch) 1', color='green')
    plt.title('Fundamental Frequency (F0) of Speech Signal 1')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.ylim(0, 500)  # 根据实际基频范围调整
    plt.legend()

    # 第二组语音信号波形图
    plt.subplot(4, 2, 4)
    plt.plot(time2, y2, label='Speech Signal 2', color='purple')
    plt.title('Speech Signal 2 (Waveform)')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()

    # 第二组语音信号包络图
    plt.subplot(4, 2, 5)
    plt.plot(time2, envelope2, label='Envelope 2', color='orange')
    plt.title('Envelope of Speech Signal 2')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()

    # 第二组语音信号基频轨迹图
    plt.subplot(4, 2, 6)
    plt.plot(time2, f0_2_interpolated, label='F0 (Pitch) 2', color='brown')
    plt.title('Fundamental Frequency (F0) of Speech Signal 2')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.ylim(0, 500)  # 根据实际基频范围调整
    plt.legend()

    # 调整布局
    plt.tight_layout()
    plt.show()

def yuputu():
    import librosa
    import librosa.display
    import numpy as np
    import matplotlib.pyplot as plt

    # ========== 1. 加载两组音频文件 ==========
    audio_file_1 = 'D:\\研究生文献\\ASV\\LA\\ASVspoof2019_LA_train\\flac\\LA_T_4345283.flac'  # 第一组音频文件路径
    '''
    1518499  7228482
    '''
    audio_file_2 = 'D:\\研究生文献\\ASV\\LA\\ASVspoof2019_LA_train\\flac\\LA_T_3559970.flac'  # 第二组音频文件路径

    y1, sr1 = librosa.load(audio_file_1, sr=None)
    y2, sr2 = librosa.load(audio_file_2, sr=None)

    # ========== 2. 计算短时傅里叶变换（STFT） ==========
    # 你可以根据需要调整 n_fft、hop_length 等参数
    D1 = librosa.stft(y1, n_fft=1024, hop_length=512, window='hann')
    D2 = librosa.stft(y2, n_fft=1024, hop_length=512, window='hann')

    # ========== 3. 将复数谱转换为幅度（分贝）谱 ==========
    S1 = librosa.amplitude_to_db(np.abs(D1), ref=np.max)
    S2 = librosa.amplitude_to_db(np.abs(D2), ref=np.max)

    # ========== 4. 绘制语谱图 ==========
    plt.figure(figsize=(12, 8))

    # 第一组语音的语谱图
    plt.subplot(2, 1, 1)
    librosa.display.specshow(S1, sr=sr1, hop_length=512, x_axis='time', y_axis='hz', cmap='magma')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram - Audio 1')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')

    # 第二组语音的语谱图
    plt.subplot(2, 1, 2)
    librosa.display.specshow(S2, sr=sr2, hop_length=512, x_axis='time', y_axis='hz', cmap='magma')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram - Audio 2')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')

    plt.tight_layout()
    plt.show()

def melspetrum():
    import librosa
    import librosa.display
    import numpy as np
    import matplotlib.pyplot as plt

    # ========== 1. 加载两组音频文件 ==========
    audio_file_1 = 'D:\\研究生文献\\ASV\\LA\\ASVspoof2019_LA_train\\flac\\LA_T_4339311.flac'  # 第一组音频文件路径
    '''
    1518499  7228482
    '''
    audio_file_2 = 'D:\\研究生文献\\ASV\\LA\\ASVspoof2019_LA_train\\flac\\LA_T_7612910.flac'  # 第二组音频文件路径

    y1, sr1 = librosa.load(audio_file_1, sr=None)
    y2, sr2 = librosa.load(audio_file_2, sr=None)

    # ========== 2. 计算梅尔频谱 ==========
    # 你可以根据需要调整 n_fft、hop_length、n_mels 等参数
    n_fft = 1024
    hop_length = 512
    n_mels = 128

    # 第一组音频
    mel_spec_1 = librosa.feature.melspectrogram(y=y1, sr=sr1, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    # 将功率谱转换为分贝刻度
    mel_spec_db_1 = librosa.power_to_db(mel_spec_1, ref=np.max)

    # 第二组音频
    mel_spec_2 = librosa.feature.melspectrogram(y=y2, sr=sr2, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    mel_spec_db_2 = librosa.power_to_db(mel_spec_2, ref=np.max)

    # ========== 3. 绘制梅尔频谱图 ==========
    plt.figure(figsize=(12, 8))

    # 第一组音频的梅尔频谱图
    plt.subplot(2, 1, 1)
    librosa.display.specshow(mel_spec_db_1,
                             sr=sr1,
                             hop_length=hop_length,
                             x_axis='time',
                             y_axis='mel',
                             cmap='magma')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel-Spectrogram - Audio 1')
    plt.xlabel('Time (s)')
    plt.ylabel('Mel Frequency')

    # 第二组音频的梅尔频谱图
    plt.subplot(2, 1, 2)
    librosa.display.specshow(mel_spec_db_2,
                             sr=sr2,
                             hop_length=hop_length,
                             x_axis='time',
                             y_axis='mel',
                             cmap='magma')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel-Spectrogram - Audio 2')
    plt.xlabel('Time (s)')
    plt.ylabel('Mel Frequency')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    print('start drawing')
    # enve()
    # pitch()
    # zcr()
    # f0()
    # env_and_f0()
    # yuputu()
    melspetrum()