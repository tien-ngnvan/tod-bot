from dm import Dialogue_Manager
dialog = ["HOTELS_1:[inform(slot4=4)]",
          "HOTELS_1:[inform(slot0=District 1)]",
          "HOTELS_1:[inform(slot8=7) and request_alts]",
          "HOTELS_1:[select]",
          "HOTELS_1:[negate_intent]",
          "HOTELS_1:[thank_you]",
          "HOTELS_1:[inform(slot3=2) and inform(slot2=March 9th) and inform(slot1=1)]",
          "HOTELS_1:[inform(slot5=CayVang) and negate]",
          "HOTELS_1:[inform(slot0=District 3)]",
          "HOTELS_1:[affirm_intent]",
          "HOTELS_1:[inform(slot3=3) and negate]",
          "HOTELS_1:[affirm and request(slot9) and request(slot8)]",
          "HOTELS_1:[thank_you]",
          ]

dm = Dialogue_Manager()

if len(dialog) >0:
    for i in range(len(dialog)):
        dm.format_string(dialog[i])
        if dm.classify_dst == "tod":
            dm.transform_action()
        else:
            print("USER:   general_asking")
            print("SYSTEM: general_asking")
        print("\n")
        if i == -1:
            while(1):
                data_dst_1 = input("Output dst: ")
                dialog.append(data_dst_1)
                dm.format_string(data_dst_1)
                if dm.classify_dst == "tod":
                    dm.transform_action()
                else:
                    print("USER:   general_asking")
                    print("SYSTEM: general_asking")
                print("\n")
else:
    while(1):
        data_dst_1 = input("Output dst: ")
        dialog.append(data_dst_1)
        dm.format_string(data_dst_1)
        if dm.classify_dst == "tod":
            dm.transform_action()
        else:
            print("USER:   general_asking")
            print("SYSTEM: general_asking")
        print("\n")