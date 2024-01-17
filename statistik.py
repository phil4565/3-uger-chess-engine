import stockfish
import chess
import chess.engine
import chess.pgn
#import christianyap
import random
import numpy as np
import math
import positions
import nn

def random_move(board):
    moves = list(board.legal_moves)
    move = random.choice(moves)
    return move

piece_values = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0
}

def evaluate_board(board):
    score = 0
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            if piece.color == chess.WHITE:
                score += piece_values[piece.piece_type]
            else:
                score -= piece_values[piece.piece_type]
    return score

def minimax(board, depth, alpha, beta, white_to_play):
    if depth == 0:
        return evaluate_board(board), None
    if white_to_play:
        best_score = -9999
        best_move = None
        for move in board.legal_moves:
            board.push(move)
            score, _ = minimax(board, depth - 1, alpha, beta, False)
            board.pop()
            if score > best_score:
                best_score = score
                best_move = move
            alpha = max(alpha, best_score)
            if alpha >= beta:
                break
        return best_score, best_move
    else:
        best_score = 9999
        best_move = None
        for move in board.legal_moves:
            board.push(move)
            score, _ = minimax(board, depth - 1, alpha, beta, True)
            board.pop()
            if score < best_score:
                best_score = score
                best_move = move
            beta = min(beta, best_score)
            if alpha >= beta:
                break
        return best_score, best_move
    
def best_move_using_minimax(board, depth):
    white_to_play = board.turn
    best_score = -99999 if white_to_play else 99999
    best_move = random_move(board)
    for move in board.legal_moves:
        board.push(move)
        score, _ = minimax(board, depth - 1, -99999, 99999, not white_to_play)
        board.pop()
        if (white_to_play and score > best_score) or (not white_to_play and score < best_score):
            best_score = score
            best_move = move
    return best_move

piece_values_with_position = {
    chess.PAWN: [
        [0, 0, 0, 0, 0, 0, 0, 0],
        [50, 50, 50, 50, 50, 50, 50, 50],
        [10, 10, 20, 30, 30, 20, 10, 10],
        [5, 5, 10, 25, 25, 10, 5, 5],
        [0, 0, 0, 20, 20, 0, 0, 0],
        [5, -5, -10, 0, 0, -10, -5, 5],
        [5, 10, 10, -20, -20, 10, 10, 5],
        [0, 0, 0, 0, 0, 0, 0, 0]
    ],
     chess.KNIGHT: [
        [-50, -40, -30, -30, -30, -30, -40, -50],
        [-40, -20, 0, 0, 0, 0, -20, -40],
        [-30, 0, 10, 15, 15, 10, 0, -30],
        [-30, 5, 15, 20, 20, 15, 5, -30],
        [-30, 0, 15, 20, 20, 15, 0, -30],
        [-30, 5, 10, 15, 15, 10, 5, -30],
        [-40, -20, 0, 5, 5, 0, -20, -40],
        [-50, -40, -30, -30, -30, -30, -40, -50]
    ],
    chess.BISHOP: [
        [-20, -10, -10, -10, -10, -10, -10, -20],
        [-10, 0, 0, 0, 0, 0, 0, -10],
        [-10, 0, 5, 10, 10, 5, 0, -10],
        [-10, 5, 5, 10, 10, 5, 5, -10],
        [-10, 0, 10, 10, 10, 10, 0, -10],
        [-10, 10, 10, 10, 10, 10, 10, -10],
        [-10, 5, 0, 0, 0, 0, 5, -10],
        [-20, -10, -10, -10, -10, -10, -10, -20]
    ],
     chess.ROOK: [
        [0, 0, 0, 0, 0, 0, 0, 0],
        [5, 10, 10, 10, 10, 10, 10, 5],
        [-5, 0, 0, 0, 0, 0, 0, -5],
        [-5, 0, 0, 0, 0, 0, 0, -5],
        [-5, 0, 0, 0, 0, 0, 0, -5],
        [-5, 0, 0, 0, 0, 0, 0, -5],
        [-5, 0, 0, 0, 0, 0, 0, -5],
        [0, 0, 0, 5, 5, 0, 0, 0]
    ],
    chess.QUEEN: [
        [-20, -10, -10, -5, -5, -10, -10, -20],
        [-10, 0, 0, 0, 0, 0, 0, -10],
        [-10, 0, 5, 5, 5, 5, 0, -10],
        [-5, 0, 5, 5, 5, 5, 0, -5],
        [0, 0, 5, 5, 5, 5, 0, -5],
        [-10, 5, 5, 5, 5, 5, 0, -10],
        [-10, 0, 5, 0, 0, 0, 0, -10],
        [-20, -10, -10, -5, -5, -10, -10, -20]
    ],
      chess.KING: [
        [-30, -40, -40, -50, -50, -40, -40, -30],
        [-30, -40, -40, -50, -50, -40, -40, -30],
        [-30, -40, -40, -50, -50, -40, -40, -30],
        [-30, -40, -40, -50, -50, -40, -40, -30],
        [-20, -30, -30, -40, -40, -30, -30, -20],
        [-10, -20, -20, -20, -20, -20, -20, -10],
        [20, 20, 0, 0, 0, 0, 20, 20],
        [20, 30, 10, 0, 0, 10, 30, 20]
    ]
}

def get_piece_value(piece, x, y):
    if piece.piece_type == chess.PAWN:
        return 100 + piece_values_with_position[piece.piece_type][x][y]
    elif piece.piece_type == chess.KNIGHT:
        return 320 + piece_values_with_position[piece.piece_type][x][y]
    elif piece.piece_type == chess.BISHOP:
        return 330 + piece_values_with_position[piece.piece_type][x][y]
    elif piece.piece_type == chess.ROOK:
        return 500 + piece_values_with_position[piece.piece_type][x][y]
    elif piece.piece_type == chess.QUEEN:
        return 900 + piece_values_with_position[piece.piece_type][x][y]
    elif piece.piece_type == chess.KING:
        return 20000 + piece_values_with_position[piece.piece_type][x][y]
    else:
        return 0


def evaluate_board(board):
    score = 0
    for i in range(8):
        for j in range(8):
            piece = board.piece_at(chess.square(i, j))
            if piece is not None:
                if piece.color == chess.WHITE:
                    score += get_piece_value(piece, i, j)
                else:
                    score -= get_piece_value(piece, i, j)

    return score

def initialize_stockfish():
    return chess.engine.SimpleEngine.popen_uci("stockfish/stockfish-windows-x86-64-modern.exe")

def stockfish_evaluation(engine, board, time_limit = 0.01):
    result = engine.analyse(board, chess.engine.Limit(time=time_limit))
    return result['score'].relative.score()





stockfish_engine = initialize_stockfish()

#board = chess.Board("rnbqkbnr/pppp1ppp/8/4p3/6P1/5P2/PPPPP2P/RNBQKBNR b KQkq - 0 2")
#result = stockfish_evaluation(stockfish_engine, board)
#print(result)

def genpos(pgnfile):
    """Save positions from pgnfile.

    Read each position in each game in pgnfile and for each position,
    generate all legal moves and save the resulting positions in epd format.
    Also save the current position.

    Sample output:
      rnb1kbnr/pp2pppp/1q1p4/8/3NP3/8/PPP2PPP/RNBQKB1R w KQkq -,42

    The 42 is the number of legal moves.
    """
    tmp = {}  # {<epd1,legalmoves>: 1, <epd2,legalmoves>: 1, ...}
    num_games = 0
    sum_scores = 0
    lsab = []
    lsstock = []
    lsdif = []
    lsnn = []
    
    with open(pgnfile, 'r') as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break

            
            for node in game.mainline():
                board = node.parent.board()
                epd = board.epd()
               
                fen = chess.Board(epd)
                scoreab = evaluate_board(fen)
                #print(scoreab)
                scorestock = stockfish_evaluation(stockfish_engine, fen)
                if scorestock is not None:
                    #print(abs(scorestock - scoreab))
                    lsdif.append(abs(scorestock - scoreab))
                #print(scorestock)
                #scorenn = nn.validate_single_position(fen)
                #print(scorenn)

                #print(lsdif)
                num_games += 1
                #print((score))  # console log
                #lsstock.append(scorestock)
                lsab.append(scoreab)
                #lsnn.append(scorenn)

                # Copy the current board and generate all moves, get
                # the epd and save it.
                tboard = board.copy()
                for m in tboard.legal_moves:
                    # Push the move and save the resulting position.
                    tboard.push(m)
                    tepd = tboard.epd()

                    # Save unique positions only.
                    legal_moves = tboard.legal_moves.count()
                    key = f'{tepd},{legal_moves}'
                    if key not in tmp:
                        tmp[key] = 1
                    tboard.pop()  # unmake and continue

                # Also save the current position.
                legal_moves = board.legal_moves.count()
                key = f'{epd},{legal_moves}'
                if key not in tmp:
                    tmp[key] = 1
    
    # Save positions in a file.
    #with open('candidates.epd', 'w') as w:
    #    for epd in list(tmp.keys()):
    #        w.write(f'{epd}\n')
                    

    print(f" Der er {num_games} games")

    #for i in range(len(lsab)):
    #    lsdif.append(abs(lsstock[i] - lsab[i]))
    
    fen_scores = []
    count = 0
    dif_snn = []

    with open(pgnfile, 'r') as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break

            for node in game.mainline():
                count += 1
                board = node.parent.board()
                epd = board.epd()
                score = nn.validate_single_position(epd)
                scorestock = stockfish_evaluation(stockfish_engine, fen)
                fen_scores.append((score))
                print(score, count)  # console log
                dif_snn.append(abs(scorestock - score))
                    
    print(fen_scores)



    #minimax
    mean = np.mean(lsab)
    sd = np.std(lsab)
    var = np.var(lsab)
    print(f"The mean of the minimax data is {mean}")
    print(f"The standard deviation of the minimax data is {sd}")
    print(f"The variance of the minimax data is {var}")

    bound_mm = 1.96 * np.sqrt(var/num_games)
    print(f"The bound of the minimax is {bound_mm}")
    #beregn 95% conf interval:
    conf_low_duberino_dif = mean - 1.96 * np.sqrt(var/num_games)
    conf_high_duberino_dif = mean + 1.96 * np.sqrt(var/num_games)
    print(f"[{mean - bound_mm} ; {mean + bound_mm}]")

    print("---------------------------------------------------")

    #nn
    mean_nn = np.mean(fen_scores)
    sd_nn = np.std(fen_scores)
    var_nn = np.var(fen_scores)
    print(f"The mean of the nn data is {mean_nn}")
    print(f"The standard deviation of the nn data is {sd_nn}")
    print(f"The variance of the nn data is {var_nn}")

    bound_nn = 1.96 * np.sqrt(var_nn/num_games)
    print(f"The bound of nn is {bound_nn}")
    #beregn 95% conf interval:
    conf_low_duberino_dif = mean_nn - 1.96 * np.sqrt(var_nn/num_games)
    conf_high_duberino_dif = mean_nn + 1.96 * np.sqrt(var_nn/num_games)
    print(f"[{mean_nn - bound_nn} ; {mean_nn + bound_nn}]")

    print("---------------------------------------------------")

    #stock - minimax
    mean_dif = np.mean(lsdif)
    sd_dif = np.std(lsdif)
    var_dif = np.var(lsdif)
    print(f"The mean of the stockfish - minimax data is {mean_dif}")
    print(f"The standard deviation of the stockfish - minimax data is {sd_dif}")
    print(f"The variance of the stockfish - minimax data is {var_dif}")


    bound_dif = 1.96 * np.sqrt(var_dif/num_games)
    print(f"The bound of stockfish - minimax is {bound_dif}")
    #beregn 95% conf interval:
    conf_low_duberino_dif = mean_dif - 1.96 * np.sqrt(var_dif/num_games)
    conf_high_duberino_dif = mean_dif + 1.96 * np.sqrt(var_dif/num_games)
    print(f"[{mean_dif - bound_dif} ; {mean_dif + bound_dif}]")

    print("---------------------------------------------------")

    #stock - nn
    mean_dif2 = np.mean(dif_snn)
    sd_dif2 = np.std(dif_snn)
    var_dif2 = np.var(dif_snn)
    print(f"The mean of the stockfish - nn data is {mean_dif2}")
    print(f"The standard deviation of the stockfish - minimax data is {sd_dif2}")
    print(f"[The variance of the dif stockfish - nn is {var_dif2}]")


    bound_dif2 = 1.96 * np.sqrt(var_dif2/num_games)
    print(f"The bound of stockfish - nn is {bound_dif2}")
    #beregn 95% conf interval:
    conf_low_duberino_dif = mean_dif2 - 1.96 * np.sqrt(var_dif2/num_games)
    conf_high_duberino_dif = mean_dif2 + 1.96 * np.sqrt(var_dif2/num_games)
    print(f"[{mean_dif2 - bound_dif2} ; {mean_dif2 + bound_dif2}]")




    

    #sumlist = sum(ls)
    #print(sumlist)
    #mean = sum(lsab)/num_games
    #print(f"Gennemsnittet er {mean}")

    #SE = 0
    #for score in lsab:
    #    SE += (score - mean)**2
    
    #MSE = SE/num_games
    #print(f"Der er {num_games} games")
    #print(SE)
    #print(f"MSE er {MSE}")
    #print(MSE**0.5)

# Start
pgnfile = 'wchcand22.pgn'
# download: https://theweekinchess.com/assets/files/pgn/wchcand22.pgn
genpos(pgnfile)

stockfish_engine.quit()