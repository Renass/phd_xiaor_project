
PROMPT = ["Go to the sphere"]
FILE_NAME = 'C:\pythonprogv2\phd_xiaor_project\sa-traj_dataset\sa-trajs2023-11-03_18-14-26_pormpt.txt' 

if __name__ == '__main__':
    with open(FILE_NAME, 'w') as file:
        for prompt in PROMPT:
            file.write(prompt + '\n')   