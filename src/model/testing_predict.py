import torch
from src.model.encoder_decoder import EncoderDecoder
from src.data.prepare_data import Data
from src.utils.shared_embedding import create_pretrained_embedding
from src.utils.device import setup_device


def load_model_for_prediction(model_path: str):
    """Load everything needed for prediction"""
    # 1. Load the saved checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')

    # 2. Recreate embedding and data processor
    shared_embedding, navec = create_pretrained_embedding(path="../embeddings/navec_hudlit_v1_12B_500K_300d_100q.tar")
    data = Data(navec)
    data.word_field.vocab = checkpoint['vocab']  # Restore vocabulary

    # 3. Recreate model architecture
    model = EncoderDecoder(
        target_vocab_size=checkpoint['model_config']['target_vocab_size'],
        shared_embedding=shared_embedding,
        d_model=checkpoint['model_config']['d_model'],
        heads_count=checkpoint['model_config']['heads_count']
    )

    # 4. Load trained weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model, data


# Usage
DEVICE = setup_device()
model, data = load_model_for_prediction("/home/nur/Documents/HW_3/model/model-1_epoch.pt")
model.to(DEVICE)

# Sample prediction
sample_text = "Пластик произвел революцию в секторе пищевой упаковки, продлив срок хранения свежих продуктов. Однако команда испанских и немецких ученых выяснила, что некоторые виды упаковки — чайные пакетики — выделяют миллионы пластиковых частиц во время использования. Чайные пакетики обычно состоят из полимолочной кислоты. Однако они могут содержать и другие виды полимеров, поэтому для анализа ученые использовали материалы различного химического состава. Два вида пакетиков для чая — нейлоновые и полипропиленовые — приобретались онлайн. Третий исследуемый тип — зеленый чай местного бренда в упаковке из целлюлозы. В результате наблюдения обнаружилось, что заваривание этих чайных пакетиков в горячей воде привело к высвобождению большого количества пластиковых наночастиц и нитевидных структур. Пакетик из полипропилена выделил около 1,2 млрд частиц на 1 мл, средний размер фрагментов составил 136,7 нанометра. Пакетик из целлюлозы выделил около 135 млн частиц на 1 мл, средний размер фрагментов составил 244 нанометра. Пакетик из нейлона выделил около 8,18 млн частиц на 1 мл, средний размер фрагментов составил 138,4 нанометра. «С помощью передовых методов нам удалось определить эти загрязнители, что важно для продвижения исследований их возможного воздействия на здоровье человека», — объяснила одна из авторов работы Альба Гарсия-Родригез из Автономного университета Барселоны. Ученые также окрасили полученные частицы, чтобы посмотреть, что происходит с нанопластиком при взаимодействии с клетками кишечника человека. Эксперименты привели к выводу, что клетки, генерирующие слизь для защиты желудочно-кишечного тракта, наиболее активно поглощали пластиковые частицы. «Крайне важно разработать стандартизированные методы испытаний, которые позволят исследовать загрязнение пластиковыми материалами, контактирующими с пищевыми продуктами. Также необходимо сформулировать нормативную политику для эффективной минимизации отходов. Поскольку использование пластика в пищевой упаковке продолжает расти, нужно решить проблему загрязнения микро- и нанопластиком, обеспечить безопасность пищевых продуктов и защитить здоровье населения», — подытожили исследователи."  # Your news text here
prediction = model.predict(
    source_text=sample_text,
    data=data,
    max_length=50,
    device=DEVICE
)
print("Generated summary:", prediction)
