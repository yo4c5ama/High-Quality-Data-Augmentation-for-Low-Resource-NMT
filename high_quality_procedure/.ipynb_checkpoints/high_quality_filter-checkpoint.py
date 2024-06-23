import kenlm
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import norm
import numpy as np


original_de = "./data/aug_double/train.de_hsb.de"
original_hsb = "./data/aug_double/train.de_hsb.hsb"

val_original_de = "../data/aug_double/val.de_hsb.de"
val_original_hsb = "../data/aug_double/val.de_hsb.hsb"

mono_de = "./data/aug_double/mono.de_hsb.de"
translations = "./data/aug_double/created.de_hsb.hsb"

TM_dehsb = "../data/transformer_L2_.8/train.dehsb_hsb.dehsb"


def read_data(path):
    with open(path, 'r', encoding="utf-8") as f:
        text = f.read()
    return text.split('\n')


def write_file(path, sentences):
    with open(path, 'w', encoding="utf-8") as f:
        for i in sentences:
            f.write(i + '\n')


def normfun(x, mu, sigma):
    function = np.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))
    return function

def normal_dist(x, mu, sigma, A):
    return A * norm.pdf(x, loc=mu, scale=sigma)

def Histograms_Normal_Distribution(ppl, path):
    mean = np.mean(ppl)
    std = np.std(ppl)
    x = np.arange(0, np.max(ppl), 0.01)
    y = normfun(x, mean, std)

    plt.plot(x, y)

    # plt.hist(ppl, bins=50, rwidth=0.7, density=False)

    # plt.title('perplexity distribution')

    plt.xlabel('Ratio (hsb/de)')
    # plt.xlabel('Perplexity')

    plt.ylabel('Probability Density')
    # plt.ylabel('Number')
    plt.savefig(path)
    plt.show()



def scatter_plot(de_ppl, hsb_ppl, path):

    plt.scatter(x=de_ppl, y=hsb_ppl, alpha=0.01)
    plt.xlabel("de_perplexity")
    plt.ylabel("hsb_perplexity")

    plt.savefig(path)
    plt.show()


def curve_fit_plot(data, path, task):
    mean = np.mean(data)
    std = np.std(data)
    hist, bins = np.histogram(data, bins=10)

    width = 0.7 * (bins[1] - bins[0])
    # patches, bins, n = plt.hist(data, bins=50, rwidth=0.7, density=False)
    center = (bins[:-1] + bins[1:]) / 2
    plt.bar(center, hist, align='center', width=width, color='C1')

    popt, pcov = curve_fit(normal_dist, center, hist, maxfev=500000)

    x = np.linspace(center[0], center[-1], 100)
    plt.plot(x, normal_dist(x, *popt), label='fit', color='C0')
    if task == "len":
        plt.xlabel("Length of German : Length of Upper Sorbian")
    if task == "ppl":
        plt.xlabel("Perplexity of German : Perplexity of Upper Sorbian")
    plt.ylabel("Number of sentences")
    plt.legend(['μ: {:.2f}, σ: {:.2f}'.format(popt[0], popt[1]), 'mean: {:.2f}, std: {:.2f}'.format(mean, std)])
    plt.savefig(path)
    plt.show()

    pass


def ppl_distribution(de, hsb):
    de_model = "../data/perplexity/orginal_bilingual_corpus/original_de.bin"
    hsb_model = "../data/perplexity/orginal_bilingual_corpus/original_hsb.bin"
    de_model = kenlm.LanguageModel(de_model)
    hsb_model = kenlm.LanguageModel(hsb_model)
    de_ppl = []
    hsb_ppl = []
    ratio_ppl = []
    flag_de = 0
    flag_hsb = 0
    for de_sen, hsb_sen in zip(de, hsb):
        de_score = de_model.perplexity(de_sen)
        hsb_score = hsb_model.perplexity(hsb_sen)
        ratio = hsb_score/de_score
        # ratio_ppl.append(ratio)
        # de_ppl.append(de_score)
        # hsb_ppl.append(hsb_score)
        # if ratio < 12:
        #     ratio_ppl.append(ratio)
        # else:
        #     flag_de += 1
        if de_score < 2500 and hsb_score < 6000:
            de_ppl.append(de_score)
            hsb_ppl.append(hsb_score)
        else:
            flag_de += 1
            flag_hsb += 1

    Histograms_Normal_Distribution(de_ppl, "original_de_perlexity_Gaussian.jpg")
    Histograms_Normal_Distribution(hsb_ppl, "original_hsb_perlexity_Gaussian.jpg")
    # # Histograms_Normal_Distribution(ratio_ppl, "original_ratio_perlexity_Histograms.jpg")
    # Histograms_Normal_Distribution(ratio_ppl, "original_ratio_perlexity_Gaussian.jpg")

    # scatter_plot(de_ppl,hsb_ppl,"created_de_hsb_perlexity.jpg")
    print(flag_de, flag_hsb)


def len_distribution(de, hsb):
    ratio_len = []
    flag = 0
    for de_sen, hsb_sen in zip(de, hsb):
        if len(de_sen) != 0 and len(hsb_sen) != 0 and len(de_sen) / len(hsb_sen) < 1.45 and len(de_sen) / len(hsb_sen) > 1.01:
            flag += 1
            ratio_len.append((len(de_sen) / len(hsb_sen)))

    curve_fit_plot(ratio_len, "filtered_length_ratio.jpg")
    print(flag)


def ppl_fit_distribution(de, hsb):
    de_model = kenlm.LanguageModel("../data/perplexity/orginal_bilingual_corpus/original_de.bin")
    hsb_model = kenlm.LanguageModel("../data/perplexity/orginal_bilingual_corpus/original_hsb.bin")
    ratio_ppl = []
    flag = 0
    for de_sen, hsb_sen in zip(de, hsb):
        de_score = de_model.perplexity(de_sen)
        hsb_score = hsb_model.perplexity(hsb_sen)
        ratio = de_score/hsb_score
        # ratio_ppl.append(ratio)

        if ratio < 5 and ratio > 0:
            ratio_ppl.append(ratio)
            flag += 1


    curve_fit_plot(ratio_ppl, "created_perlexity_ratio.jpg")


    print(flag)


def filter_according_length(de, hsb):
    de_model = kenlm.LanguageModel("./data/perplexity/orginal_bilingual_corpus/original_de.bin")
    hsb_model = kenlm.LanguageModel("./data/perplexity/orginal_bilingual_corpus/original_hsb.bin")
    ratio_len = []
    ratio_ppl = []
    flag = 0
    for de_sen, hsb_sen in zip(de, hsb):
        if len(de_sen) != 0 and len(hsb_sen) != 0 and len(de_sen) / len(hsb_sen) < 1.45 and len(de_sen) / len(hsb_sen) > 1.01:
            de_score = de_model.perplexity(de_sen)
            hsb_score = hsb_model.perplexity(hsb_sen)
            ratio = de_score / hsb_score
            if ratio < 10:
                ratio_ppl.append(ratio)
                flag += 1

            # ratio_ppl.append(ratio)
            # flag += 1


    curve_fit_plot(ratio_ppl, "filtered_ppl_ratio_by_length.jpg")
    print(flag)


def filter_distribution(de, hsb, ppl_up=3, ppl_down=0, len_up=3, len_down=0):
    ratio_len = []
    ratio_ppl = []
    de_filtered = []
    hsb_filtered = []
    de_model = kenlm.LanguageModel("./data/perplexity/orginal_bilingual_corpus/original_de.bin")
    hsb_model = kenlm.LanguageModel("./data/perplexity/orginal_bilingual_corpus/original_hsb.bin")
    flag = 0
    for de_sen, hsb_sen in zip(de, hsb):
        de_score = de_model.perplexity(de_sen)
        hsb_score = hsb_model.perplexity(hsb_sen)
        ratio = de_score / hsb_score
        if de_sen != '' and hsb_sen != '':
            len_ratio = len(de_sen) / len(hsb_sen)
            # if 1.49 > ratio > 0.63 and 3 > len_ratio > 0:
            # if 1.49 > ratio > 0.63 and 1.35 > len_ratio > 1.01:
            # if 5 > ratio > 0 and 1.35 > len_ratio > 1.01:
            # if 5 > ratio > 0 and 3 > len_ratio > 0:
            if ppl_up > ratio > ppl_down and len_up > len_ratio > len_down:
                ratio_ppl.append(ratio)
                ratio_len.append(len_ratio)
                de_filtered.append(de_sen)
                hsb_filtered.append(hsb_sen)
                flag += 1
    # print(flag)
    return de_filtered, hsb_filtered, ratio_len, ratio_ppl




def TM_len_distribution(dehsb):
    dehsb_len = []


    flag = 0
    for sentence in dehsb:
        if len(sentence) < 125:

            dehsb_len.append(len(sentence))
        else:
            flag += 1

    curve_fit_plot(dehsb_len, "TM_dehsb_len.jpg", "len")
    # curve_fit_plot(ratio_ppl, "filtered_ppl_ratio_by_double.jpg", "ppl")
    print(flag)


def filter_synthetic(ppl_up, ppl_down, len_up, len_down):
    de = read_data(original_de)
    hsb = read_data(original_hsb)
    de_filtered, hsb_filtered, ratio_len, ratio_ppl = filter_distribution(de, hsb, ppl_up, ppl_down, len_up, len_down)
    write_file("./data/aug_double/filtered.de_hsb.de", de_filtered)
    write_file("./data/aug_double/filtered.de_hsb.hsb", hsb_filtered)
    curve_fit_plot(ratio_len, "./figure/filtered_length_ratio_by_double.pdf", "len")
    curve_fit_plot(ratio_ppl, "./figure/filtered_ppl_ratio_by_double.pdf", "ppl")


def get_original_interval():
    de = read_data(original_de)
    hsb = read_data(original_hsb)
    de_filtered, hsb_filtered, ratio_len, ratio_ppl = filter_distribution(de, hsb)
    curve_fit_plot(ratio_len, "./figure/original_length_ratio.pdf", "len")
    curve_fit_plot(ratio_ppl, "./figure/original_perlexity_ratio.pdf", "ppl")
    return np.mean(ratio_ppl), np.std(ratio_ppl), np.mean(ratio_len), np.std(ratio_len)

if __name__ == '__main__':

    de = read_data(original_de)
    hsb = read_data(original_hsb)

    # de = read_data(val_original_de)
    # hsb = read_data(val_original_hsb)

    # de = read_data(mono_de)
    # hsb = read_data(translations)

    # dehsb = read_data(TM_dehsb)
    # ppl_distribution(de, hsb)
    # len_distribution(de, hsb)
    # ppl_fit_distribution(de, hsb)
    # filter_according_length(de, hsb)
    filter_distribution(de, hsb)
    # TM_len_distribution(de)
    pass


