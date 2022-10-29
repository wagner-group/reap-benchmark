import pandas as pd
import os
import argparse

def get_args_parser():
    parser = argparse.ArgumentParser(description='Save images for manual annotation', add_help=False)
    parser.add_argument('--column', default='todo', type=str)
    parser.add_argument('--group', default=1, type=int)
    return parser

def main(args, corrections_df):
    # df = pd.read_csv('../../../../data/shared/mtsd_v2_fully_annotated/traffic_sign_annotation_train.csv')
    # df = pd.read_csv('../../../../data/shared/mtsd_v2_fully_annotated/traffic_sign_annotation_validation.csv')
    # df = pd.read_csv('error_df_validation.csv')
    
    # column = args.column
    
    circle_filename_list = ['mex_9F6W0gjVgB4_vmUDnA_70.png', 'fyQOwxNP4PY-ThFgoSG5gw_34.png',
       'fyQOwxNP4PY-ThFgoSG5gw_68.png', 'MfAWvyhxKPJMrQ4z73K06g_53.png',
       'kpynFI07L7VQEGmiIuauPw_73.png', 'xtSkSjIA8O3ee666K4CVIQ_13.png',
       'xtSkSjIA8O3ee666K4CVIQ_14.png', 'e6ZS57GUlfgphaLk7WGNkQ_172.png',
       'e6ZS57GUlfgphaLk7WGNkQ_179.png', 'zryllJQGwxq6g23RuSB-ag_32.png',
       'zryllJQGwxq6g23RuSB-ag_33.png', 'gZcBxlYXzMTXB-9cwrRiig_8.png',
       'gZcBxlYXzMTXB-9cwrRiig_15.png', 'fX-Bdr_iPUkUybIYjtaZZA_100.png',
       'fAsCW7xDAvjamZqWIp1J3g_57.png', 'fAsCW7xDAvjamZqWIp1J3g_58.png',
       'aGWqpdOWr9Uq6pjoYi_wSg_10.png', 'z3xp88xBjag_KwsMglNZyw_123.png',
       'yIX71NWG8WZhakeJqGrxgA_42.png', 'c0NHwONSQ1xpQRItUmGSdQ_23.png',
       'pBBmjuJ8yU1r2ROYRzWmFg_202.png', 'nR-M2zUbIWJzatAuy2egrQ_22.png',
       'sRD7CHhGB1dpKU2QSUgyyQ_127.png', 'h5ZD44k4ZgQ0INd6S7u3fQ_19.png',
       'nJ07SmokAFtGwa9lA2YBZA_10.png', 'nJ07SmokAFtGwa9lA2YBZA_12.png',
       'nMdximodlz7gqfJDM_pvgQ_68.png', 'AXCXuA5b-pXNIT22DyPg0w_82.png',
       'AXCXuA5b-pXNIT22DyPg0w_86.png', 'AXCXuA5b-pXNIT22DyPg0w_87.png',
       'AXCXuA5b-pXNIT22DyPg0w_91.png', 'AXCXuA5b-pXNIT22DyPg0w_94.png',
       'C3fao_-9rHFlTCldCc_neQ_26.png', 'C3fao_-9rHFlTCldCc_neQ_70.png',
       'BOhRWGdo7UI6jKJ67rDw8Q_45.png', '-CjGcO8hfpC8lzGrYsM8Iw_47.png',
       '--WOpVBuHlCygAUADkttpg_70.png', '--WOpVBuHlCygAUADkttpg_72.png',
       '0Vsc8UPOf6xf-IKjZSG8wA_29.png', '0Vsc8UPOf6xf-IKjZSG8wA_59.png',
       '6obAf9CRQh_dBFHPAIiRFQ_125.png', '6mlTHzDsv69RDAVp7bMNbA_101.png',
       '3PZpBha4oNlwpGbc_HwDNA_19.png', '3PZpBha4oNlwpGbc_HwDNA_104.png',
       'RA00IkVIBZ7xwXYJKqRN5A_83.png', 'RA00IkVIBZ7xwXYJKqRN5A_131.png',
       'PnQCZsOv-TJZbyY-vwOdGw_97.png', 'Sim8FB_C2VJ_y8Q0ReYUDA_47.png',
       'Sim8FB_C2VJ_y8Q0ReYUDA_51.png', 'T-GxvAalhtHL_8Ab1NRGvw_61.png',
       'T-GxvAalhtHL_8Ab1NRGvw_74.png', 'IR32Ac4-HM6ZODjXZQzdaA_49.png',
       'IR32Ac4-HM6ZODjXZQzdaA_56.png', 'NhxFEhVrAwV6Vso2JL9Oig_92.png',
       'p4yMkbdRQDBc1S5Uw7C7eg_53.png', 'p4yMkbdRQDBc1S5Uw7C7eg_55.png',
       'ZFMhdiH0fKKK4jxqOJK_zw_76.png']

    triangle_filename_list = ['aR_1PNPVMM6hvyaRAlyAEg_61.png', 'xKUId2cijkTWZ0akhrPDKQ_80.png',
       'xkc3223mXUZZXiD597-eRw_74.png', 'uEVeAIICDqLCsEPbzMBo_w_83.png',
       'UK7t19FMgwvNeluFwy3tOw_37.png', 'FiBH3zmdR4X36jz2UlG1tA_27.png',
       'LJTXcs6VrjdfjHZT_y7Nng_18.png']

    inverted_triangle_filename_list = ['hTra1szcXD3O2j9JF-PfJQ_72.png', 'Cpg5lWF9Ui0l3PXaH_zVEQ_43.png',
       '3vhQjI8NQelkS28IyvYH2g_72.png', '3HipILLiorEPR4nUmO6kqA_31.png',
       'PxjKHcsORBAjJUHtpwMdrw_95.png', 'VI1jE3jEntJ-ndBnxhhMdw_104.png',
       'WKTrRKpi14B2HQoJ4VAcMA_52.png', 'T8pOTPpaN6ky92BQsiFuxg_24.png',
       'U8ihSYp0H8FxbazvJZYr3g_60.png', 'K5la755JyNbj9j8skvW2mA_225.png']

    diamond_600_filename_list = ['0KohgmStOYkLZM6v-Frfew_96.png', 'eEQ3dQ0UU1Kf14z81WknqA_24.png',
       'xoDewWfyWyn5OImKZOJ0QA_27.png', 'YRijNSbYEB5H9GMz_M5X2g_67.png',
       'yZqd0dueHhb0PcDU0bmmWA_117.png', 't7OhttxBEIOTwxUvv3gC4w_44.png',
       'rncW3BXgi5tMpJZM7G2a1Q_42.png', 'hAztO_Ch_qPltDZriaap0A_111.png',
       'hqd7rzXiPVGuXY9VUV2fbg_108.png', 'ww0Amqxgn3VIeAsCz50sBQ_35.png',
       'm7oNMb8b5hWhNGtf6hKxEQ_122.png', 'llFwZLOsOUnQLr-qyFWBcw_27.png',
       'AZIL6FY41M1BPSA_qBUzAg_44.png', '7b3_ODjZjnBxtqNEZOQxuQ_105.png',
       '2avKZZ-CcNQjtiU4835Y_w_78.png', 'QKUDFO5ccJliPSjmv2bf-w_65.png',
       'LFUzFqltNVweCmW7LFJBdQ_16.png']

    diamond_915_filename_list = ['jlzyNKfTd4bfJNh4U_fGCw_47.png', 'rolz92wuhIfjkAWmTf6QjQ_78.png',
       'NpUFZoQQ8qTL0ug-L4zgKA_39.png', 'ermsC5h4ga5ZyRFPeTNFCg_38.png',
       'e9QE7DkwK1h5yjW_CQhmeQ_108.png', 'dOn-65n9ojjQxxtQ-XgK4Q_39.png',
       'acYZZIs3QoY60afgKCpYuQ_59.png', 'yQMHm4taGHzNFXtDbe0K6g_107.png',
       'camoHp_U2GRE7I_Beba9yg_125.png', 'sTs_8E2qnFxJuMFKYAmcHw_57.png',
       'jNqP-shFuacamGVkM0rTRw_29.png', 'n4cu9x_p1aYPjzYGPwUdww_117.png',
       'kRBmCTeXOyrN8wJMudDS9Q_43.png', 'EgCFwjZguLbkKdlfl8s0BQ_98.png',
       '1cWSOsWhN5C3yfM1i6vVTg_28.png', '2zvLDLPqBLiVzc6sPnTbtg_116.png',
       '-qv7yT8igffux_xn1j38_g_98.png', 'R0LF2x6W3BjkEC_DuX4BKA_70.png',
       'R5fEksuOgLLZvaxSJGg1Tw_81.png', 'SKypS9o2AZ3Em37Kg2t4xQ_36.png',
       'PC3ohlzXEcVrDhMwpBMBfA_60.png', 'WhAfU1Ytz7KNyIuNGmgrHg_16.png',
       'GH6N12hepCPbA_cWArjgfg_74.png']

    square_filename_list = ['5CzcwmVX6jqOdvUI2--6xw_121.png', 'bcIMxYCNO0s2-9EEqP3eQg_134.png',
       'wT178UNZ3bbG6mUBGWUpXg_46.png', 'xXpwsbuYSmRwfdJ0rr1Liw_62.png',
       '_S3gC5NzN9scKX2RFlS8kg_78.png', 'z2QB55VTB-8sHBX65kH6_Q_106.png',
       'm3mgfoWD_8Rs5T5E00-xnA_97.png', 'm50WiNHHC_aNJRQh5gNdlw_45.png',
       'jqMNbLC01AGBFMJyElxLjw_151.png', '9ngARyO6zFzYapvyrbdHYQ_37.png',
       '-BqO16ocxK46wM5W-QCE_A_60.png', '5X_nZuTXZlLoh9VXWjzVqA_86.png',
       'Ocqxx-dKKW3_yFK-6gKUKA_43.png', 'P2h7xzLy_qsq4eyAP2iAcw_90.png',
       'VnrrDSrqiYSpzg-__ETWXA_176.png', 'WcpFA74dbeBIpwx9DRubCg_9.png',
       'W4Uj_HomlZRc2udiFnoWrw_50.png', 'UjyMm6w_ojsPuszxP8oy4A_61.png',
       'Iu5b_rfCLDcswfUaou1cLQ_62.png', 'HS6OJq4d4NjDzD8pjccRgg_16.png',
       'H1umQpLlBL-N8nZO0b6CIA_73.png', 'KPjiyMxQ2fIYHv4qdYt-NQ_94.png',
       'p4yMkbdRQDBc1S5Uw7C7eg_47.png']

    rect_458_filename_list = ['qxJ9jILGLeJyCsLjQpUzAQ_47.png', 'akl9UAP1LAwx7mJTGY94qw_19.png',
       'CqHCnUa75cedqgh0dS0IPA_43.png', 'B8us7fIegpDX1usB_WGI8Q_79.png',
       '8TOsQa-UguNueNYqkNd4cw_11.png', 'ypb1zSpAJZ1l_MpSKKP8pg_64.png',
       '_JeXVytV1OD0R7HSI9NAXQ_24.png', 'qF7SkRmiz_SBMKon3SHWmQ_24.png',
       'rbPNqyb8NHL6CsIAaJsCBg_34.png', 'ijl3G75djRlMJvCLbnpHqg_73.png',
       'voU9Y1Lr0VMiTerTJ0OTtA_89.png', 'mMpduF5xFcTB8siwqCOfkg_66.png',
       'n0rmCMKzstnvyH5tipDnFg_30.png', 'DP2RS4z_7e1fht5Sveda-A_130.png',
       'DP2RS4z_7e1fht5Sveda-A_131.png', 'EQjzhx2ZXF4KSQ87U2iRGA_92.png',
       'C1piGGdcyu9S2axCrGnY-w_46.png', 'BT3G22uY8-VmRqC0WAwA7Q_75.png',
       'DL7oeLvLynzS6FMn6yvdLw_130.png', 'CUFulVNJePu89iZ_dcyHZw_14.png',
       '1Teb9w_ZQXmRtpTBDEYNWQ_46.png', 'VBDKYOJUZibL6AcgkYXsGw_52.png',
       'HkygmP6JKtYU6_0fAIPCCA_82.png']

    rect_762_filename_list = ['buHNdH2-X-DtVD3SZodkgw_50.png', '4CO00pCAZELnIh8bmb23ig_185.png',
       'ypb1zSpAJZ1l_MpSKKP8pg_71.png', 'teUMoHGn42s8BTg7xdFkWw_114.png',
       'tSq4yEdwuxHnrrC96byxYg_123.png', 'rOfzfNACM76ymQcrjuEEAA_94.png',
       'goNbG6U6UXja9p6Us5xAnA_103.png', 'goNbG6U6UXja9p6Us5xAnA_104.png',
       'Dax5hN4gArn2Q9VlVFmblA_22.png', 'EQjzhx2ZXF4KSQ87U2iRGA_93.png',
       'BjMI2XSzVoYm0mSzw54mDw_111.png', 'CNe8aGc64gXQJSIXkHChgA_65.png',
       '4fSZuK9Ix_fzwBFoLxaSnQ_244.png', '46P_SNQCh9OfgyPRsRj6oA_38.png',
       'V7mE9UIDgzvpPGa4Bb7NCA_72.png', 'IZRnApBIRiohGH1X96Rkhg_34.png',
       'KFoovPrd8d7MV-CQVc8tDg_70.png']

    rect_915_filename_list = ['Uo21MGB_hwwTXI9z1Txu2A_68.png', 'YjrHeKbJw29xWYOHAfZeiQ_31.png',
       'ZOo4iYq7CI96h3-quFJQYw_95.png', 'q6ACuij3PjVlUYfLkyTd9w_76.png',
       'kkZdf49kZfUFkUGbbY-yoA_45.png', 'AXoUqnE2XNxxJMskNL8Ykw_37.png',
       '-R2IhI2eSfE85UcN4nladg_43.png', '0zk_b7xzXVZugRUyu7E2Gg_51.png',
       '0SPjG7loAzo5lVMCktr8PQ_42.png', 'RbPDI6VkIPE29sbY7SWwKQ_90.png']

    pentagon_filename_list = ['bbfNQHuYLehPra3l-WZJRg_167.png', 'qV87USRRsFcKm5edOpirIQ_113.png',
       'jePRuRtnGYQLTE7T1pErAw_74.png', '7I1i-KJyxONKd7tGL8Dw-g_102.png',
       '57RX6_BDC8sYop4gU8DpUw_68.png']

    octagon_filename_list = ['8CvaYyTKsEQ6lI8DysjvyQ_91.png', 'ZMXbEmXXI50ITAC-XWeC3A_42.png',
       'yeMrH_vVD72OEjS6-qC3TA_66.png', 'qFW7U1TuNeWgOOg7zhBFSA_64.png',
       'o8SoPF8vAqS2QCPGAzhuQQ_91.png', '8DIhvJHkdY2vgnvppGq3Jw_41.png',
       'TJ_7Td9UyfbwnVeoZJkSGA_29.png', 'Ifa87jBAQV4lYyJF1tx1Zw_35.png',
       'ML4Mc_KzhbzsUh_TXncHQw_63.png', 'N1Wmvfpqc5Ip8ZKDg9GAEw_36.png']

    filename_list = circle_filename_list + triangle_filename_list + inverted_triangle_filename_list +\
                    diamond_600_filename_list + diamond_915_filename_list + square_filename_list +\
                    rect_458_filename_list + rect_762_filename_list + rect_915_filename_list +\
                    pentagon_filename_list + octagon_filename_list


    column = 'patch_errors'
    # df = df[df['occlusion'].isna()]
    # df = df[df[f'{column}'] == 1]
    # df = df[df['group'] == args.group]

    # corrections_df = pd.concat([corrections_df, df], axis=0)
    
    print('number of images to label', len(filename_list), '\n')
    # qqq
    # print('number of images to label', len(df), '\n')
    split = 'training'
    # data_dir = '/data/shared/mapillary_vistas/training/'
    data_dir = f'/data/shared/mapillary_vistas/{split}/'

    img_path_cropped = os.path.join(data_dir, 'traffic_signs')
    img_path = os.path.join(data_dir, 'images')

    for filename in filename_list:
    # for filename in df['filename'].values:
        new_filename = '_'.join(filename.split('_')[:-1]) + '.jpg'

        img_file_cropped = os.path.join(img_path_cropped, filename)
        img_file = os.path.join(img_path, new_filename)

        # img_file_destination = f'/data/shared/mapillary_vistas/training/traffic_signs_{column}/{args.group}/'
        # img_file_destination_cropped = f'/data/shared/mapillary_vistas/training/traffic_signs_{column}/{args.group}_cropped/'
        img_file_destination = f'/data/shared/mapillary_vistas/{split}/traffic_signs_{column}/{args.group}/'
        img_file_destination_cropped = f'/data/shared/mapillary_vistas/{split}/traffic_signs_{column}/{args.group}_cropped/'

        if not os.path.exists(img_file_destination):
            os.makedirs(img_file_destination)

        if not os.path.exists(img_file_destination_cropped):
            os.makedirs(img_file_destination_cropped)

        # if os.path.isfile(img_file) and os.path.isfile(os.path.join(os.path.join('/data/shared/mapillary_vistas/training/', 'images'), new_filename)):
        #     print()
        #     print(filename)
        #     print(new_filename)
        #     print(img_file)
        #     qqqq

        if not os.path.isfile(img_file):
            # print(filename)
            # print(new_filename)
            # print(img_file)
            # raise Exception()
            continue

        img_file_destination += filename
        img_file_destination_cropped += filename
        os.system(f'cp {img_file} {img_file_destination}') 
        os.system(f'cp {img_file_cropped} {img_file_destination_cropped}') 

    # corrections_df.to_csv("mapillary_vistas_corrections.csv", index=False)
    # corrections_df.to_csv("mapillary_vistas_corrections_validation.csv", index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Save images for manual annotation', parents=[get_args_parser()])
    args = parser.parse_args()

    try:
        corrections_df = pd.read_csv("mapillary_vistas_corrections.csv")
        # corrections_df = pd.read_csv("mapillary_vistas_corrections_validation.csv")
    except:
        corrections_df = pd.DataFrame()

    main(args, corrections_df)