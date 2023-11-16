class Colors:
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    RESET = "\033[0m"


def debug():

    print(Colors.RED + "RED" + Colors.RESET)
    print(Colors.GREEN + "GREEN" + Colors.RESET)
    print(Colors.YELLOW + "YELLOW" + Colors.RESET)
    print(Colors.BLUE + "BLUE" + Colors.RESET)
    print(Colors.MAGENTA + "MAGENTA" + Colors.RESET)
    print(Colors.CYAN + "CYAN" + Colors.RESET)
    print(Colors.WHITE + "WHITE" + Colors.RESET)

    for i in range(30, 38):
        for j in range(40, 48):
            print(f"\033[{i};{j}m TextColor:{i} BgColor:{j} \033[0m", end=" ")
        print()

    # Bright colors
    for i in range(90, 98):
        for j in range(100, 108):
            print(f"\033[{i};{j}m TextColor:{i} BgColor:{j} \033[0m", end=" ")
        print()
