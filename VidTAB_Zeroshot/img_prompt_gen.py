import os


def read_text(data_path):
    label_map_path = f"{data_path}/label_map.txt"
    with open(label_map_path, 'r') as file:
        data_str = file.read()
    file.close()
    label_map = eval(data_str)

    label_candidate = []

    keys = sorted([int(key) if isinstance(key, str) else key for key in label_map.keys()])
    all_str = all(isinstance(key, str) for key in label_map.keys())
    label_candidate = [label_map[str(key)] if all_str else label_map[key] for key in keys]

    return prompt_generate(data_path, label_candidate)


def prompt_generate(data_path, label_candidate):
    if 'arid' in data_path.lower():
        label_prompt = []

        reformat = {"Drink": "Drinking",
                    "Jump": "Jumping",
                    "Pick": "Picking",
                    "Pour": "Pouring",
                    "Push": "Pushing",
                    "Run": "Running",
                    "Sit": "Sitting",
                    "Stand": "Standing",
                    "Turn": "Turning",
                    "Walk": "Walking",
                    "Wave": "Waving", }

        for label in label_candidate:
            label_prompt.append(f'Someone is {reformat[label]}.')
    elif 'breakfast' in data_path.lower():
        label_prompt = []
        for label in label_candidate:
            label_prompt.append(f'A video of making {label}.')
    elif 'surgicalactions' in data_path.lower():
        label_prompt = []
        for label in label_candidate:
            label_prompt.append(f'Performing {label} in gynecologic laparoscopy.'.replace('_', ' ').replace('-', ' '))
    elif 'facefake' in data_path.lower():
        label_prompt = []
        for label in label_candidate:
            if label == "video with fake face":
                label_prompt.append(f'A video of a manipulated face.')
            elif label == "origin video":
                label_prompt.append(f'A video of a natural face.')
            else:
                raise NotImplementedError
    elif 'caer' in data_path.lower():
        label_prompt = []
        for label in label_candidate:
            label_prompt.append(f'Someone is {label.lower()}.')  # "A person is showing sadness."
    elif 'dover' in data_path.lower():
        label_prompt = []
        for label in label_candidate:
            #label_prompt.append(f'A {label}.')
            if 'low' in label.lower():
                label_prompt.append(f'A video of low quality.')
            elif 'high' in label.lower():
                label_prompt.append(f'A video of high quality.')
    elif 'mob' in data_path.lower():
        label_prompt = []
        for label in label_candidate:
            if 'without' in label.lower():
                label_prompt.append("Harmless content.")
            elif 'violent' in label.lower():
                label_prompt.append("Fast, repetitive or violent actions.")
            elif "unpleasant" in label.lower():
                label_prompt.append("Obscene or unpleasant cartoon content.")
    elif 'animal_kingdom' in data_path.lower():
        label_prompt = []
        for label in label_candidate:
            label_prompt.append(f"The animal is {label['Action']}.".rstrip('\n'))
    else:
        raise NotImplementedError

    return label_prompt


if __name__ == '__main__':
    ROOT = "/mnt/petrelfs/lixinhao/lxh_exp/data/video_eval"

    datasets = ['animal_kingdom', 'breakfast', 'MOB',
                'SurgicalActions160', 'CAER', 'DOVER',
                'facefake', 'ARID']

    for data in datasets:
        print(os.path.join(ROOT, data))
        print(read_text(os.path.join(ROOT, data)))
