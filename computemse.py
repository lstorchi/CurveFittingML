import sys
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":

    uselog10 = False
    line1 = -1
    line2 = -2
    if len(sys.argv) != 5:
        print("Usage: python3 script.py <filename> line1 line2 log10=T,F")
        sys.exit(1)

    uselog10 = sys.argv[4].strip().lower() == 't'
    line1 = int(sys.argv[2])
    line2 = int(sys.argv[3])
    filename = sys.argv[1]

    try:
        with open(filename, 'r') as file:
            yp = []
            yt = []
            lines = file.readlines()
            for line in lines:
                sline = line.split(",")
                if len(sline) <= 2:
                    print(f"Error: Each line must contain exactly two values separated by a comma.")
                    sys.exit(1)
                try:
                    yp.append(float(sline[line1]))
                    yt.append(float(sline[line2]))
                except ValueError:
                    print(f"Error: Non-numeric data found in the file jump line.")

            if len(yp) != len(yt):
                print(f"Error: The number of predicted and true values do not match.")
                sys.exit(1)
            if uselog10:
                xvalue = np.arange(len(yp))
                yp = np.array(yp)
                yt = np.array(yt)
                yp =np.log10(yp)
                yt =np.log10(yt)
                mse = sum((yp[i] - yt[i]) ** 2 for i in range(len(yp))) / len(yp)
                print(f"Mean Squared Error (log10): {mse}")
                rmse = mse ** 0.5
                print(f"Root Mean Squared Error (log10): {rmse}")
                plt.plot(xvalue, yp, 'r', label='Predicted')
                plt.plot(xvalue, yt, 'b', label='True')
                plt.xlabel('Index')
                plt.ylabel('log10 Value')
                plt.title('Predicted vs True Values (log10)')
                plt.legend()
                #plt.show()
                plt.savefig("log10.png")
                plt.close()
            else:
                mse = sum((yp[i] - yt[i]) ** 2 for i in range(len(yp))) / len(yp)
                print(f"Mean Squared Error: {mse}")
                rmse = mse ** 0.5
                print(f"Root Mean Squared Error: {rmse}")
                xvalue = np.arange(len(yp))
                plt.plot(xvalue, yp, 'r', label='Predicted')
                plt.plot(xvalue, yt, 'b', label='True')
                plt.xlabel('Index')
                plt.ylabel('Value')
                plt.title('Predicted vs True Values')
                plt.legend()
                #plt.show()
                plt.savefig("mse.png")
                plt.close()
    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found.")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)