import math

AlexCapsNet = [83.16,	80.86,	84.13,	72.49,	77.64,	74.72,	72.49,	83.96,	74.8,	83.41,
               72.58,	79.02,	75.39,	72.42,	74.73,	82.58,	72.09,	76.43,	83.43,	73.65,
               76.13,	81.46,	74.05,	76.83,	72.14,	83.6,	76.24,	73.55,	83.14,	76.03,
               73.49,	68.91,	83.12,	72.43,	82.77,	82.21,	73.41,	74.48,	79.32,	71.44,
               75.68,	71.64,	71.59,	83.12,	75.44,	73.35,	83.51,	82.67,	76.19,	81.88
    ]
CapNet = [71.16, 70.54, 70.64, 71.11, 70.33, 69.34, 71.39, 71.05, 71.39, 71.19,
          70.70, 70.83, 70.72, 71.63, 70.96, 71.17, 71.09, 70.26, 71.18, 71.21,
          71.04, 71.62, 71.51, 70.57, 71.33, 70.9,  70.22, 71.27, 71.05, 71.75,
          70.72, 71.17, 70.63, 71.28, 70.52, 70.85, 71.25, 70.8,  70.93, 70.8,
          70.11, 70.78, 70.64, 71.43, 68.58, 71.1,  69.69, 68.71, 70.63, 71.41
      ]
AlexNet = [68.41, 67.16, 72.33, 73.52, 74.36, 68.66, 73.96, 72.28, 71.8, 74.25,
      74.84, 74.24, 70.37, 67.65, 73.59, 67.26, 70.32, 70.07, 73.65, 74.35,
      72.61, 66.97, 72.83, 74.21, 73.84, 76,    74.2,  73.98, 74.88, 74.93,
      74.51, 74.39, 73.09, 74.66, 74.87, 69.63, 72.36, 73.75, 72.54, 74.75,
      73.21, 72.91, 74.21, 70.24, 73.05, 71.37, 73.43, 73.44, 74.97, 70.98
]

# def calculate(ACN, CN):
#     N = len(ACN)
#     avg_ACN = sum(ACN) / len(ACN)
#     avg_CN  = sum(CN) / len(CN)
#
#     # delta
#     delta = [a - b for a, b in zip(ACN, CN)]
#     # print(delta)
#     delta_avg = sum(delta) / N
#     print(f"delta_avg is {delta_avg}")
#
#     # S
#     # print([(x-delta_avg) for x in delta])
#     S2 = sum([(x-delta_avg)**2 for x in delta]) / (N-1)
#     # print(f"S2 is {S2}")
#     S = math.sqrt(S2)
#     print(f'S is {S}')
#
#     # Delta_avg/S
#     DeltaAvg_div_S = delta_avg / S
#     print(f'final score is {DeltaAvg_div_S}')
#     print(f'T is {DeltaAvg_div_S*math.sqrt(50)}')


def calculatev2(ACN, CN):
    N = len(ACN)
    avg_ACN = sum(ACN) / len(ACN)
    avg_CN  = sum(CN) / len(CN)
    print(f'average accuracy of ACN is {avg_ACN}')
    print(f'average accuracy of CN is {avg_CN}')
    print(f'avg_ACN-avg_CN is {avg_ACN-avg_CN}')

    S2_ACN = sum([(x-avg_ACN)**2 for x in ACN]) / N
    print(f'S2_ACN is {S2_ACN}')
    S2_CN  = sum([(x-avg_CN) ** 2 for x in CN]) / N
    print(f'S2_CN is {S2_CN}')

    area = math.sqrt(S2_ACN/N + S2_CN/N)
    print(f'area is {area}')


print('---------------------------------ACN-CN------------------------')
#
calculatev2(AlexCapsNet, CapNet)
print('---------------------------------ACN-AN------------------------')
calculatev2(AlexCapsNet, AlexNet)





