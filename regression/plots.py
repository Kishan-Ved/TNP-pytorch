# import re
# import matplotlib.pyplot as plt

# def parse_log_file(filepath):
#     steps = []
#     losses = []

#     # Regex pattern to match log lines and extract step and loss
#     pattern = re.compile(r"step (\d+).*?loss ([\d.]+)")

#     with open(filepath, 'r') as file:
#         for line in file:
#             line = line.strip()
#             if not line:
#                 continue  # skip blank lines

#             match = pattern.search(line)
#             if match:
#                 step = int(match.group(1))
#                 loss = float(match.group(2))
#                 steps.append(step)
#                 losses.append(loss)

#     return steps, losses

# def plot_loss_vs_steps(steps, losses):
#     plt.figure(figsize=(10, 6))
#     plt.plot(steps, losses, marker='o', linestyle='-', color='teal', label='Train Loss')
#     plt.title('Training Loss vs Steps')
#     plt.xlabel('Step')
#     plt.ylabel('Loss')
#     plt.grid(True)
#     plt.legend()
#     plt.tight_layout()
#     plt.show()

# if __name__ == "__main__":
#     log_file = "your_log_file.txt"  # Replace with your actual log filename
#     steps, losses = parse_log_file(log_file)
#     plot_loss_vs_steps(steps, losses)








import re
import matplotlib.pyplot as plt
import seaborn as sns

# Pretty style
sns.set(style="whitegrid", context="talk", font_scale=1.1)

def parse_log_file(filepath):
    steps = []
    losses = []
    tar_lls = []

    # Handles optional negative sign and scientific notation
    pattern = re.compile(r"step (\d+).*?loss ([\d.eE+-]+).*?tar_ll (-?[\d.eE+-]+)")

    with open(filepath, 'r') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue

            match = pattern.search(line)
            if match:
                steps.append(int(match.group(1)))
                losses.append(float(match.group(2)))
                tar_lls.append(float(match.group(3)))

    return steps, losses, tar_lls

def plot_loss_and_tarll(steps, losses, tar_lls, save_path=None):
    if not steps:
        print("⚠️ No data found in the log file. Check the format or path.")
        return

    plt.figure(figsize=(12, 6))

    # Plotting both curves
    # plt.plot(steps, losses, label='Loss', color='teal', marker='o')
    steps = steps[1:]
    tar_lls = tar_lls[1:]
    print(tar_lls)
    plt.plot(steps, tar_lls, label='tar_ll', color='darkorange', marker='s')

    # Titles and labels
    plt.title("Log-Likelihood vs Steps")
    plt.xlabel("Training Step")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"✅ Plot saved to {save_path}")
    else:
        plt.show()

if __name__ == "__main__":
    log_file = "/home/kishan.ved/TNP-pytorch/regression/results/gp/merged_attn/merged_2-self-attention-test/train_20250407-1613.log"  # Replace with your actual file
    steps, losses, tar_lls = parse_log_file(log_file)
    plot_loss_and_tarll(steps, losses, tar_lls)
