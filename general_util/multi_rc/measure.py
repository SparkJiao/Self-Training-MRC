import math


class Measures:

    @staticmethod
    def per_question_metrics(dataset, output_map):
        P = []
        R = []
        # for p in dataset:
        #     for qIdx, q in enumerate(p["paragraph"]["questions"]):
        #         id = p["id"] + "==" + str(qIdx)
        #         if (id in output_map):
        #             predictedAns = output_map.get(id)
        #             correctAns = [int(a["isAnswer"]) for a in q["answers"]]
        #             predictCount = sum(predictedAns)
        #             correctCount = sum(correctAns)
        #             assert math.ceil(sum(predictedAns)) == sum(predictedAns), "sum of the scores: " + str(sum(predictedAns))
        #             agreementCount = sum([a * b for (a, b) in zip(correctAns, predictedAns)])
        #             p1 = (1.0 * agreementCount / predictCount) if predictCount > 0.0 else 1.0
        #             r1 = (1.0 * agreementCount / correctCount) if correctCount > 0.0 else 1.0
        #             P.append(p1)
        #             R.append(r1)
        #         else:
        #             print("The id " + id + " not found . . . ")
        for article_id, article in dataset.items():
            for question_id, question in article.items():
                predicted_ans = Measures.get_sorted_list(output_map[article_id][question_id])
                correct_ans = Measures.get_sorted_list(question)
                predict_count = sum(predicted_ans)
                correct_count = sum(correct_ans)
                assert math.ceil(sum(predicted_ans)) == sum(predicted_ans), "sum of the scores" + str(sum(predicted_ans))
                agreement_count = sum([a * b for (a, b) in zip(correct_ans, predicted_ans)])
                p1 = (1.0 * agreement_count / predict_count) if predict_count > 0.0 else 1.0
                r1 = (1.0 * agreement_count / correct_count) if correct_count > 0.0 else 1.0
                P.append(p1)
                R.append(r1)

        pAvg = Measures.avg(P)
        rAvg = Measures.avg(R)
        f1Avg = 2 * Measures.avg(R) * Measures.avg(P) / (Measures.avg(P) + Measures.avg(R))
        return [pAvg, rAvg, f1Avg]

    @staticmethod
    def exact_match_metrics(dataset, output_map, delta):
        EM = []
        # for p in dataset:
        #     for qIdx, q in enumerate(p["paragraph"]["questions"]):
        #         id = p["id"] + "==" + str(qIdx)
        #         if (id in output_map):
        #             predictedAns = output_map.get(id)
        #             correctAns = [int(a["isAnswer"]) for a in q["answers"]]
        #             em = 1.0 if sum([abs(i - j) for i, j in zip(correctAns, predictedAns)]) <= delta  else 0.0
        #             EM.append(em)
        #         else:
        #             print("The id " + id + " not found . . . ")
        for article_id, article in dataset.items():
            for question_id, question in article.items():
                predicted_ans = Measures.get_sorted_list(output_map[article_id][question_id])
                correct_ans = Measures.get_sorted_list(question)
                em = 1.0 if sum(abs(i - j) for i, j in zip(correct_ans, predicted_ans)) <= delta else 0.0
                EM.append(em)

        return Measures.avg(EM)

    @staticmethod
    def per_dataset_metric(dataset, output_map):
        agreementCount = 0
        correctCount = 0
        predictCount = 0
        # for p in dataset:
        #     for qIdx, q in enumerate(p["paragraph"]["questions"]):
        #         id = p["id"] + "==" + str(qIdx)
        #         if (id in output_map):
        #             predictedAns = output_map.get(id)
        #             correctAns = [int(a["isAnswer"]) for a in q["answers"]]
        #             predictCount += sum(predictedAns)
        #             correctCount += sum(correctAns)
        #             agreementCount += sum([a * b for (a, b) in zip(correctAns, predictedAns)])
        #         else:
        #             print("The id " + id + " not found . . . ")
        for article_id, article in dataset.items():
            for question_id, question in article.items():
                predict_ans = Measures.get_sorted_list(output_map[article_id][question_id])
                correct_ans = Measures.get_sorted_list(question)
                predictCount += sum(predict_ans)
                correctCount += sum(correct_ans)
                agreementCount += sum([a * b for (a, b) in zip(correct_ans, predict_ans)])

        p1 = (1.0 * agreementCount / predictCount) if predictCount > 0.0 else 1.0
        r1 = (1.0 * agreementCount / correctCount) if correctCount > 0.0 else 1.0
        return [p1, r1, 2 * r1 * p1 / (p1 + r1)]

    @staticmethod
    def avg(l):
        # return reduce(lambda x, y: x + y, l) / len(l)
        return 1.0 * sum(l) / len(l)

    @staticmethod
    def get_sorted_list(dic):
        sorted_dic = sorted(dic.items(), key=lambda x: x[0], reverse=False)
        return [x[1] for x in sorted_dic]
