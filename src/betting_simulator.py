from .mongoFootballClient import MongoFootballClient
from .config import Config as conf
from tqdm import tqdm
from src.data_models.result import Result

class BettingSimulator:
    def __init__(self, year, silly=False):
        self.mfc = MongoFootballClient(conf.MONGO_URL)
        self.year_to_evaluate = year
        self.odds = self.mfc.get_odds(year)
        self.predictions = self.mfc.get_predictions(year)
        self.silly = silly
        self.run_simulation()

    def run_simulation(self):
        print(f"Running simulation for year {self.year_to_evaluate}")
        bankroll = 0
        lowest_bankroll = 0
        bets_made = []
        correct_results = 0
        wrong_results = 0
        predictions = 0
        diffs = []
        odds = []
        bet_chances = 0
        total_bet = 0
        for odd in tqdm(self.odds):
            if bankroll < lowest_bankroll:
                lowest_bankroll = bankroll
            prediction = self.mfc.get_prediction(odd.date, odd.home_team)
            odd.home_odds = odd.home_odds * 0.95
            odd.away_odds = odd.away_odds * 0.95
            odd.draw_odds = odd.draw_odds * 0.95
            
            if prediction is not None and (odd.date, odd.home_team) not in bets_made:
                bet_chances += 1
                result = self.mfc.get_match_result(odd.date, odd.home_team)
                bets_made.append((odd.date, odd.home_team))
                odd_home_prob = 1/odd.home_odds
                odd_away_prob = 1/odd.away_odds
                odd_draw_prob = 1/odd.draw_odds

                home_diff = prediction.home_win - odd_home_prob
                away_diff = prediction.away_win - odd_away_prob
                draw_diff = prediction.draw - odd_draw_prob
                max_diff = max(home_diff, away_diff, draw_diff)
                diffs.append(max_diff)

                if self.silly:
                    predictions += 1
                    odds.append(odd.home_odds)
                    total_bet += 1
                    if result.value == "Home Win":
                        #print(f"Home Win {odd.date}, {home_team}, {odd.home_odds}")
                        bankroll += odd.home_odds-1
                        correct_results += 1
                    else:
                        bankroll -= 1
                        wrong_results += 1
                else:
                    if home_diff == max_diff and max_diff > 0.15:
                        #home_team = self.mfc.get_team_from_id(odd.home_team)
                        predictions += 1
                        odds.append(odd.home_odds)
                        if result.value == "Home Win":
                            #print(f"Home Win {odd.date}, {home_team}, {odd.home_odds}")
                            bankroll += odd.home_odds-1
                            correct_results += 1
                        else:
                            bankroll -= 1
                            wrong_results += 1
                    elif away_diff == max_diff and max_diff > 0.15:
                        #home_team = self.mfc.get_team_from_id(odd.home_team)
                        predictions += 1
                        odds.append(odd.away_odds)
                        if result.value == "Away Win":
                            #print(f"Away Win {odd.date}, {home_team}, {odd.away_odds}")
                            bankroll += odd.away_odds-1
                            correct_results += 1
                        else:
                            bankroll -= 1
                            wrong_results +=1
                    elif draw_diff > 0.15:
                        #home_team = self.mfc.get_team_from_id(odd.home_team)
                        predictions += 1
                        odds.append(odd.draw_odds)
                        if result.value == "Draw":
                            #print(f"Draw {odd.date}, {home_team}, {odd.draw_odds}")
                            bankroll += odd.draw_odds-1
                            correct_results +=1
                        else:
                            bankroll -= 1
                            wrong_results += 1
                    else:
                        pass
        print("Finished Bankroll:", bankroll)
        print("Number of bets available", bet_chances)
        print("Number of bets", predictions)
        print("Accuracy is:", correct_results/predictions)
        print("Wrong is:", wrong_results/predictions)
        print("Average best diff", sum(diffs)/len(diffs))
        print("Average odds:", sum(odds)/len(odds))
        print("Lowest Bankroll Ever got", lowest_bankroll)
        print("Total money bet:", total_bet)
        



