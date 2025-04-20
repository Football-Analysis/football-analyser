from .mongoFootballClient import MongoFootballClient
from .config import Config as conf
from tqdm import tqdm
from src.data_models.result import Result

class BettingSimulator:
    def __init__(self, year, silly=False, initial_bankroll=100, threshold=0.175, bet_percentage=0.02):
        self.mfc = MongoFootballClient(conf.MONGO_URL)
        self.year_to_evaluate = year
        self.odds = self.mfc.get_odds(year)
        self.predictions = self.mfc.get_predictions(year)
        self.silly = silly
        self.bankroll = initial_bankroll
        self.threshold = threshold
        self.amount_bet = 0
        self.bet_percentage = bet_percentage
        self.run_simulation()

    def run_simulation(self):
        print(f"Running simulation for year {self.year_to_evaluate}")
        lowest_bankroll = self.bankroll
        bets_made = []
        correct_results = 0
        wrong_results = 0
        self.predictions = 0
        diffs = []
        odds = []
        bet_chances = 0
        total_bet = 0
        for odd in tqdm(self.odds):
            if self.bankroll < lowest_bankroll:
                lowest_bankroll = self.bankroll
            prediction = self.mfc.get_prediction(odd.date, odd.home_team)
            result = self.mfc.get_match_result(odd.date, odd.home_team)
            
            if prediction is not None and (odd.date, odd.home_team) not in bets_made and result:
                bet_chances += 1
                #bet_size = min(40,self.bankroll*self.bet_percentage)

                bets_made.append((odd.date, odd.home_team))
                odd_home_prob = 1/odd.home_odds
                odd_away_prob = 1/odd.away_odds
                odd_draw_prob = 1/odd.draw_odds

                home_diff = prediction.home_win - odd_home_prob
                away_diff = prediction.away_win - odd_away_prob
                draw_diff = prediction.draw - odd_draw_prob
                max_diff = max(home_diff, away_diff, draw_diff)
                min_diff = min(home_diff, away_diff, draw_diff)

                #bet_size = min(40,self.bankroll*self.bet_percentage)
                
                if max_diff > 0.25:
                    test=""
                    bet_size = min(40,self.bankroll*0.02)
                elif max_diff > 0.2:
                    test="bad"
                    bet_size = min(40,self.bankroll*0.01)
                elif max_diff > 0.175:
                    test = "good"
                    bet_size = min(40,self.bankroll*0.005)
                else:
                    test="bad"
                    bet_size = 0


                if abs(max_diff) > abs(min_diff):
                    back = True
                else:
                    back = True

                diffs.append(max_diff)

                if self.silly:
                    self.predictions += 1
                    odds.append(odd.home_odds)
                    total_bet += 1
                    if result.value == "Home Win":
                        #print(f"Home Win {odd.date}, {home_team}, {odd.home_odds}")
                        self.bankroll += (((odd.home_odds-1)*bet_size)*0.98)
                        correct_results += 1
                    else:
                        self.bankroll -= bet_size
                        wrong_results += 1
                
                else:
                    if ((home_diff == max_diff and back) or (home_diff == min_diff and not back)) and (max_diff > self.threshold):
                        #print(test)
                        #home_team = self.mfc.get_team_from_id(odd.home_team)
                        #print(bet_size)
                        self.amount_bet += bet_size
                        if back:
                            odds.append(odd.home_odds)
                            if result.value == "Home Win" and home_diff == max_diff and home_diff > self.threshold:
                                #print(home_diff, away_diff, draw_diff)
                                self.predictions += 1
                                self.bankroll += ((odd.home_odds-1)*bet_size)*0.98
                                correct_results += 1
                            elif result.value != "Home Win" and home_diff == max_diff and max_diff > self.threshold:
                                #print(home_diff, away_diff, draw_diff)
                                self.predictions += 1
                                self.bankroll -= bet_size
                                wrong_results += 1
                        else:
                            odds.append(1+(1/(odd.home_odds-1)))
                            if result.value != "Home Win" and home_diff == min_diff and abs(min_diff) > self.threshold:
                                #print(f"Home Win {odd.date}, {home_team}, {odd.home_odds}")
                                #print(home_diff, away_diff, draw_diff)
                                back_odds = 1+(1/(odd.home_odds-1))
                                self.predictions += 1
                                self.bankroll += ((back_odds-1)*bet_size)*0.98
                                correct_results += 1
                            elif result.value == "Home Win" and home_diff == min_diff and abs(min_diff) > self.threshold:
                                #print(home_diff, away_diff, draw_diff)
                                self.predictions += 1
                                self.bankroll -= bet_size
                                wrong_results += 1
                    elif ((away_diff == max_diff and back) or (away_diff == min_diff and not back)) and (max_diff > self.threshold):
                        #home_team = self.mfc.get_team_from_id(odd.home_team)
                        #print(test)
                        self.amount_bet += bet_size
                        if back:
                            odds.append(odd.away_odds)
                            if result.value == "Away Win" and away_diff == max_diff and max_diff > self.threshold:
                                self.predictions += 1
                                self.bankroll += ((odd.away_odds-1)*bet_size)*0.98
                                correct_results += 1
                            elif result.value != "Away Win" and away_diff == max_diff and max_diff > self.threshold:
                                self.predictions += 1
                                self.bankroll -= bet_size
                                wrong_results += 1
                        else:
                            odds.append(1+(1/(odd.away_odds-1)))
                            if result.value != "Away Win" and away_diff == min_diff and abs(min_diff) > self.threshold:
                                self.predictions += 1
                                back_odds = 1+(1/(odd.away_odds-1))
                                self.bankroll += ((back_odds-1)*bet_size)*0.98
                                correct_results += 1
                            elif result.value == "Away Win" and away_diff == min_diff and abs(min_diff) > self.threshold:
                                self.predictions += 1
                                self.bankroll -= bet_size
                                wrong_results += 1
                    elif ((draw_diff == max_diff and back) or (draw_diff == min_diff and not back)) and (max_diff > self.threshold):
                        #home_team = self.mfc.get_team_from_id(odd.home_team)
                        #print(test)
                        self.amount_bet += bet_size
                        if back:
                            odds.append(odd.home_odds)
                            if result.value == "Draw" and draw_diff == max_diff and max_diff > self.threshold:
                                self.predictions += 1
                                self.bankroll += ((odd.draw_odds-1)*bet_size)*0.98
                                correct_results += 1
                            elif result.value != "Draw" and draw_diff == max_diff and max_diff > self.threshold:
                                self.predictions += 1
                                self.bankroll -= bet_size
                                wrong_results += 1
                        else:
                            odds.append(1+(1/(odd.draw_odds-1)))
                            if result.value != "Draw" and draw_diff == min_diff and abs(min_diff) > self.threshold:
                                self.predictions += 1
                                back_odds = 1+(1/(odd.draw_odds-1))
                                self.bankroll += ((back_odds-1)*bet_size)*0.98
                                correct_results += 1
                            elif result.value == "Draw" and draw_diff == min_diff and abs(min_diff) > self.threshold:
                                self.predictions += 1
                                self.bankroll -= bet_size
                                wrong_results += 1
                    else:
                        pass

            else:
                pass
                # if (odd.date, odd.home_team) in bets_made:
                #     print(f"CAnnot find prediction relating to {odd.date} {odd.home_team} {odd.away_team} - duplicate")
                # else:
                #     print(f"CAnnot find prediction relating to {odd.date} {odd.home_team} {odd.away_team} - cup")

        sorted_odds = sorted(odds)
        print("Finished bankroll:", self.bankroll)
        print("Number of bets available", bet_chances)
        print("Number of bets", self.predictions)
        print("Accuracy is:", correct_results/self.predictions)
        print("Wrong is:", wrong_results/self.predictions)
        print("Average best diff", sum(diffs)/len(diffs))
        print("Average odds:", sum(odds)/len(odds))
        print("Median odds:", sorted_odds[len(odds)//2])
        print("Lowest bankroll Ever got", lowest_bankroll)
        print("Total money bet:", total_bet)
        



