from .mongoFootballClient import MongoFootballClient
from .config import Config as conf
from tqdm import tqdm
from src.data_models.result import Result
from typing import List
from time import time

class BettingSimulator:
    def __init__(self, year, silly=False, initial_bankroll=100, threshold=0.175, bet_percentage=0.01, bankroll_history: List[float]=[]):
        self.mfc = MongoFootballClient(conf.MONGO_URL)
        self.year_to_evaluate = year
        self.odds = self.mfc.get_odds(year)
        self.predictions = self.mfc.get_predictions(year)
        self.silly = silly
        self.bankroll = initial_bankroll
        self.threshold = threshold
        self.amount_bet = 0
        self.bet_percentage = bet_percentage
        self.bankroll_history = bankroll_history
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
        back_bets = 0
        lay_bets = 0
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

                if abs(min_diff) > abs(max_diff):
                    back = False
                else:
                    back = True

                diffs.append(max_diff)

                bet_size = min(40,self.bankroll*self.bet_percentage)
                threshold = self.threshold
                if self.silly:
                    self.predictions += 1
                    odds.append(odd.home_odds)
                    total_bet += 1
                    if result.value == "Home Win":
                        self.bankroll += ((odd.home_odds-1)*bet_size)*0.98
                        correct_results += 1
                    else:
                        self.bankroll -= bet_size
                        wrong_results += 1
                
                else:
                    if (home_diff == max_diff and max_diff > threshold and odd.home_odds and back) and odd.home_odds < 35 and odd.home_odds > 1.05:# or (odd.home_odds > 50 and home_diff == max_diff and max_diff > 0.2):# < 50:
                        if odd.home_odds > 50:
                            bet_size = 1
                        back_bets += 1
                        self.amount_bet += bet_size
                        odds.append(odd.home_odds)
                        if result.value == "Home Win":
                            self.predictions += 1
                            self.bankroll += ((odd.home_odds-1)*bet_size)*0.98
                            self.bankroll_history.append(self.bankroll)
                            correct_results += 1
                        elif result.value != "Home Win":
                            self.predictions += 1
                            self.bankroll -= bet_size
                            self.bankroll_history.append(self.bankroll)
                            wrong_results += 1
                        else:
                            raise RuntimeError(f"Result {result.value} should not exist, backing Home Win")
                    elif (away_diff == max_diff and max_diff > threshold and odd.away_odds and back) and odd.away_odds < 35 and odd.away_odds > 1.05:# or (odd.away_odds > 50 and away_diff == max_diff and max_diff > 0.2):# < 50:
                        if odd.away_odds > 50:
                            bet_size = 1
                        self.amount_bet += bet_size
                        back_bets += 1
                        odds.append(odd.away_odds)
                        if result.value == "Away Win":
                            self.predictions += 1
                            self.bankroll += ((odd.away_odds-1)*bet_size)*0.98
                            self.bankroll_history.append(self.bankroll)
                            correct_results += 1
                        elif result.value != "Away Win":
                            self.predictions += 1
                            self.bankroll -= bet_size
                            self.bankroll_history.append(self.bankroll)
                            wrong_results += 1
                        else:
                            raise RuntimeError(f"Result {result.value} should not exist, backing Away Win")
                    elif (draw_diff == max_diff and max_diff > self.threshold and odd.draw_odds and back) and odd.draw_odds < 35 and odd.draw_odds > 1.05:# or (odd.draw_odds > 50 and draw_diff == max_diff and max_diff > 0.2):# < 50:
                        if odd.draw_odds > 50:
                            bet_size = 1
                        self.amount_bet += bet_size
                        back_bets += 1
                        odds.append(odd.draw_odds)
                        if result.value == "Draw":
                            self.predictions += 1
                            self.bankroll += ((odd.draw_odds-1)*bet_size)*0.98
                            self.bankroll_history.append(self.bankroll)
                            correct_results += 1
                        elif result.value != "Draw":
                            self.predictions += 1
                            self.bankroll -= bet_size
                            self.bankroll_history.append(self.bankroll)
                            wrong_results += 1
                    elif (home_diff == min_diff and max_diff > threshold and odd.home_odds and not back) and odd.home_odds < 35 and odd.home_odds > 1.05:# or (odd.home_odds > 50 and home_diff == max_diff and max_diff > 0.2):# < 50:
                        if odd.home_odds > 50:
                            bet_size = 1
                        self.amount_bet += bet_size
                        lay_bets += 1
                        back_odd = 1+(1/(odd.home_odds-1))
                        odds.append(back_odd)
                        if result.value != "Home Win":
                            self.predictions += 1
                            self.bankroll += ((back_odd-1)*bet_size)*0.98
                            self.bankroll_history.append(self.bankroll)
                            correct_results += 1
                        elif result.value == "Home Win":
                            self.predictions += 1
                            self.bankroll -= bet_size
                            self.bankroll_history.append(self.bankroll)
                            wrong_results += 1
                        else:
                            raise RuntimeError(f"Result {result.value} should not exist, laying Home Win")
                    elif (away_diff == min_diff and max_diff > threshold and odd.away_odds and not back) and odd.away_odds < 35 and odd.away_odds > 1.05:# or (odd.away_odds > 50 and away_diff == max_diff and max_diff > 0.2):# < 50:
                        if odd.away_odds > 50:
                            bet_size = 1
                        self.amount_bet += bet_size
                        lay_bets += 1
                        back_odd = 1+(1/(odd.away_odds-1))
                        odds.append(back_odd)
                        if result.value != "Away Win":
                            self.predictions += 1
                            self.bankroll += ((back_odd-1)*bet_size)*0.98
                            self.bankroll_history.append(self.bankroll)
                            correct_results += 1
                        elif result.value == "Away Win":
                            self.predictions += 1
                            self.bankroll -= bet_size
                            self.bankroll_history.append(self.bankroll)
                            wrong_results += 1
                        else:
                            raise RuntimeError(f"Result {result.value} should not exist, backing Away Win")
                    elif (draw_diff == min_diff and max_diff > self.threshold and odd.draw_odds and not back) and odd.draw_odds < 35 and odd.draw_odds > 1.05:# or (odd.draw_odds > 50 and draw_diff == max_diff and max_diff > 0.2):# < 50:
                        if odd.draw_odds > 50:
                            bet_size = 1
                        self.amount_bet += bet_size
                        lay_bets += 1
                        back_odd = 1+(1/(odd.draw_odds-1))
                        odds.append(back_odd)
                        if result.value != "Draw":
                            self.predictions += 1
                            self.bankroll += ((back_odd-1)*bet_size)*0.98
                            self.bankroll_history.append(self.bankroll)
                            correct_results += 1
                        elif result.value == "Draw":
                            self.predictions += 1
                            self.bankroll -= bet_size
                            self.bankroll_history.append(self.bankroll)
                            wrong_results += 1
                    else:
                        pass
            else:
                pass

        sorted_odds = sorted(odds)
        print("Finished bankroll:", self.bankroll)
        print("Number of bets available", bet_chances)
        print("Number of bets", self.predictions)
        print("Number of back bets", back_bets)
        print("Number of lay bets", lay_bets)
        print("Accuracy is:", correct_results/self.predictions)
        print("Wrong is:", wrong_results/self.predictions)
        print("Average best diff", sum(diffs)/len(diffs))
        print("Average odds:", sum(odds)/len(odds))
        print("Median odds:", sorted_odds[len(odds)//2])
        print("Lowest bankroll Ever got", lowest_bankroll)
        print("Total money bet:", total_bet)
        



