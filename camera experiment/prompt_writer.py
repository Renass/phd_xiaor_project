
PROMPT = ["Go through the pink gates. Avoid touching the obstacles."]
FILE_NAME = '/home/renas/pythonprogv2/phd_xiaor_project/sa-traj_dataset/real_pink_gates/prompt.txt' 

if __name__ == '__main__':
    with open(FILE_NAME, 'w') as file:
        for prompt in PROMPT:
            file.write(prompt + '\n')   