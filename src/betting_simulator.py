from .mongoFootballClient import MongoFootballClient
from .config import Config as conf
from tqdm import tqdm
from src.data_models.result import Result

class BettingSimulator:
    def __init__(self, year, silly=False, initial_bankroll=100, threshold=0.175, bet_percentage=0.01):
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

                diffs.append(max_diff)

                # if home_diff==max_diff:
                #     sizing_refactor = 3/odd.home_odds
                #     if odd.home_odds < 2.5:
                #         bet_size = min(40,self.bankroll*self.bet_percentage*1)
                #         threshold = self.threshold
                #     else:
                #         threshold = self.threshold
                #         bet_size = min(40,self.bankroll*self.bet_percentage)
                # elif away_diff==max_diff:
                #     sizing_refactor = 3/odd.away_odds
                #     if odd.away_odds < 2.5:
                #         bet_size = min(40,self.bankroll*self.bet_percentage*1)
                #         threshold = self.threshold
                #     else:
                #         threshold = self.threshold
                #         bet_size = min(40,self.bankroll*self.bet_percentage)
                # elif draw_diff == max_diff:
                #     sizing_refactor = 3/odd.draw_odds
                #     if odd.draw_odds < 2.5:
                #         bet_size = min(40,self.bankroll*self.bet_percentage*1)
                #         threshold = self.threshold
                #     else:
                #         threshold = self.threshold
                #         bet_size = min(40,self.bankroll*self.bet_percentage)

                # confidence = (max_diff - self.threshold)*100
                # confidence_adjustment = 1+(confidence*0.2)
                # pre_bet_size = self.bet_percentage*sizing_refactor*confidence_adjustment
                # #pre_bet_size=self.bet_percentage
                # bet_size = min(40,self.bankroll*pre_bet_size)

                bet_size = min(40,self.bankroll*self.bet_percentage)
                #bet_size = self.bankroll*self.bet_percentage
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
                    if (home_diff == max_diff and max_diff > threshold and odd.home_odds) or (odd.home_odds > 50 and home_diff == max_diff and max_diff > 0.2):# < 50:
                        if odd.home_odds > 50:
                            bet_size = 1
                        self.amount_bet += bet_size
                        odds.append(odd.home_odds)
                        if result.value == "Home Win":
                            self.predictions += 1
                            self.bankroll += ((odd.home_odds-1)*bet_size)*0.98
                            correct_results += 1
                        elif result.value != "Home Win":
                            self.predictions += 1
                            self.bankroll -= bet_size
                            wrong_results += 1
                        else:
                            raise RuntimeError(f"Result {result.value} should not exist, backing Home Win")
                    elif (away_diff == max_diff and max_diff > threshold and odd.away_odds) or (odd.away_odds > 50 and away_diff == max_diff and max_diff > 0.2):# < 50:
                        if odd.away_odds > 50:
                            bet_size = 1
                        self.amount_bet += bet_size
                        odds.append(odd.away_odds)
                        if result.value == "Away Win":
                            self.predictions += 1
                            self.bankroll += ((odd.away_odds-1)*bet_size)*0.98
                            correct_results += 1
                        elif result.value != "Away Win":
                            self.predictions += 1
                            self.bankroll -= bet_size
                            wrong_results += 1
                        else:
                            raise RuntimeError(f"Result {result.value} should not exist, backing Away Win")
                    elif (draw_diff == max_diff and max_diff > self.threshold and odd.draw_odds) or (odd.draw_odds > 50 and draw_diff == max_diff and max_diff > 0.2):# < 50:
                        if odd.draw_odds > 50:
                            bet_size = 1
                        self.amount_bet += bet_size
                        odds.append(odd.draw_odds)
                        if result.value == "Draw":
                            self.predictions += 1
                            self.bankroll += ((odd.draw_odds-1)*bet_size)*0.98
                            correct_results += 1
                        elif result.value != "Draw":
                            self.predictions += 1
                            self.bankroll -= bet_size
                            wrong_results += 1
                    else:
                        pass
            else:
                pass

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
        



