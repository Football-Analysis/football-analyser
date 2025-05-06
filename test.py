from src.predictor import FootballPredictor
from src.betting_simulator import BettingSimulator

#fp = FootballPredictor()
#fp.create_and_evaluate()
edge_thresholds = [0.25, 0.275, 0.3, 0.325]
sizing_thresholds = [0.005, 0.0075, 0.01, 0.0125, 0.015]
end_bankrolls = []
bet_totals = []

# for threshold in edge_thresholds:
#     total_bets = 0
#     bs= BettingSimulator("2016", threshold=threshold)
#     bankroll = bs.bankroll
#     total_bets += bs.predictions
#     bs= BettingSimulator("2017", initial_bankroll=bankroll, threshold=threshold)
#     bankroll = bs.bankroll
#     total_bets += bs.predictions
#     bs= BettingSimulator("2018", initial_bankroll=bankroll, threshold=threshold)
#     bankroll = bs.bankroll
#     total_bets += bs.predictions
#     bs= BettingSimulator("2019", initial_bankroll=bankroll, threshold=threshold)
#     bankroll = bs.bankroll
#     total_bets += bs.predictions
#     bs= BettingSimulator("2020", initial_bankroll=bankroll, threshold=threshold)
#     bankroll = bs.bankroll
#     total_bets += bs.predictions
#     bs= BettingSimulator("2021", initial_bankroll=bankroll, threshold=threshold)
#     bankroll = bs.bankroll
#     total_bets += bs.predictions
#     bs= BettingSimulator("2022", initial_bankroll=bankroll, threshold=threshold)
#     bankroll = bs.bankroll
#     total_bets += bs.predictions
#     bs= BettingSimulator("2023", initial_bankroll=bankroll, threshold=threshold)
#     bankroll = bs.bankroll
#     total_bets += bs.predictions
#     bs= BettingSimulator("2024", initial_bankroll=bankroll, threshold=threshold)
#     bankroll = bs.bankroll
#     total_bets += bs.predictions

#     end_bankrolls.append(bankroll)
#     bet_totals.append(total_bets)


#     print("TOTAL BETS MADE OVER 9 YEARS:", total_bets)
#     print("TOTAL BETTING RESULTS OVER 9 YEARS:", bankroll)

# max_bankroll = max(end_bankrolls)
# best_threshold = edge_thresholds[end_bankrolls.index(max_bankroll)]
# print(f"Biggest bankroll is with edge threshold of: {best_threshold}")
# print(f"Biggest bankroll achieved was: {max_bankroll}")
# print(f"Number of bets placed to achieve that bankroll was {bet_totals[end_bankrolls.index(max_bankroll)]}")



total_bets = 0
amount_bet = 0
bs= BettingSimulator("2016", threshold=0.3, initial_bankroll=2000, bet_percentage=0.03)
bankroll = bs.bankroll
total_bets += bs.predictions
amount_bet += bs.amount_bet
bs= BettingSimulator("2017", initial_bankroll=bankroll, threshold=0.3, bet_percentage=0.03)
bankroll = bs.bankroll
total_bets += bs.predictions
amount_bet += bs.amount_bet
bs= BettingSimulator("2018", initial_bankroll=bankroll, threshold=0.3, bet_percentage=0.03)
bankroll = bs.bankroll
total_bets += bs.predictions
amount_bet += bs.amount_bet
bs= BettingSimulator("2019", initial_bankroll=bankroll, threshold=0.3, bet_percentage=0.03)
bankroll = bs.bankroll
total_bets += bs.predictions
amount_bet += bs.amount_bet
bs= BettingSimulator("2020", initial_bankroll=bankroll, threshold=0.3, bet_percentage=0.03)
bankroll = bs.bankroll
total_bets += bs.predictions
amount_bet += bs.amount_bet
bs= BettingSimulator("2021", initial_bankroll=bankroll, threshold=0.3, bet_percentage=0.03)
bankroll = bs.bankroll
total_bets += bs.predictions
amount_bet += bs.amount_bet
bs= BettingSimulator("2022", initial_bankroll=bankroll, threshold=0.3, bet_percentage=0.03)
bankroll = bs.bankroll
total_bets += bs.predictions
amount_bet += bs.amount_bet
bs= BettingSimulator("2023", initial_bankroll=bankroll, threshold=0.3, bet_percentage=0.03)
bankroll = bs.bankroll
total_bets += bs.predictions
amount_bet += bs.amount_bet
bs= BettingSimulator("2024", initial_bankroll=bankroll, threshold=0.3, bet_percentage=0.03)
bankroll = bs.bankroll
total_bets += bs.predictions
amount_bet += bs.amount_bet


print(f"Total bankroll is {bankroll}")
print(f"Bets made {total_bets}")
print(f"Total money gambled {amount_bet}")

