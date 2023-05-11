import numpy as np
import scipy
import matplotlib.pyplot as plt
import os
from scipy import signal, fft

# генерация сигнала
n = 1000  # количество сгенерированных элементов
a = 0  # среднее распределения
b = 10  # стандартное отклонение распределения

signal = np.random.normal(a, b, n)

# настройка фильтра
n_reports = 500  # количество отсчетов
Fs = 1000  # частота дискретизации
F_max = 3  # максимальная частота повидомлений

time = np.arange(n) / Fs

# Нормирование частоты
w = F_max / (Fs/2)

# расчет параметров/коэффициентов фильтра
parameters_filter = scipy.signal.butter(3, w, 'low', output='sos')

# фильтрация сигнала
filtered_signal = scipy.signal.sosfiltfilt(parameters_filter, signal)

# функция для построения графика
def plot_graph(x, y, title, xlabel, ylabel):
    # Преобразование размера из см в дюймы
    width = 21 / 2.54
    height = 14 / 2.54

    # Создание объектов фигуры и оси заданного размера
    fig, ax = plt.subplots(figsize=(width, height))

    # Данные графика с заданной шириной линии
    ax.plot(x, y, linewidth=1)

    # Установите метки осей с указанным размером шрифта
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)

    # Установить заголовок с указанным размером шрифта
    plt.title(title, fontsize=14)

    # Установите пределы x и y для увеличения сигнала с F_max = 3
    ax.set_xlim([0, 1])
    ax.set_ylim([-15, 15])

    # Сохранить фигуру с указанным dpi
    fig.savefig('./figures/' + title + '.png', dpi=600)

# создание папки, на всякий случай
if not os.path.exists('./figures'):
    os.makedirs('./figures')

# построение графиков сигналов
plot_graph(time, signal, 'Original signal', 'Time (s)', 'Amplitude')
plot_graph(time, filtered_signal, 'Сигнал с максимальной частотой F_max = 3 (Hz)', 'Time (s)', 'Amplitude')

# расчет спектра сигнала
spectrum = scipy.fft.fft(filtered_signal)

# сдвиг нулевой частоты к центру спектра и расчет модуля спектра
spectrum_shifted = np.abs(scipy.fft.fftshift(spectrum))

# расчет частотных отсчетов спектра
freqs = scipy.fft.fftfreq(n, 1/Fs)

# сдвиг частотных отсчетов к центру спектра
freqs_shifted = scipy.fft.fftshift(freqs)

# построение графика спектра сигнала
plot_graph(freqs_shifted, spectrum_shifted, 'Спектр фильтра сигнала', 'Frequency (Hz)', 'Amplitude')
