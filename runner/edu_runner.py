from education import blockdetection, categorization, pattern_matching, preprocessing


def edu_main():
    print('Preprocessing')
    preprocessing.main()
    print('Categorization')
    categorization.main()
    print('Blockdetection')
    blockdetection.main()
    print('Pattern_matching')
    pattern_matching.main()


if __name__ == '__main__':
    edu_main()