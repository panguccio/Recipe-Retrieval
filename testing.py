import cProfile, pstats
from index import InvertedIndex
from corpus_parser import load_corpus

corpus = load_corpus("recipes/test.json")

profiler = cProfile.Profile()
profiler.enable()

index = InvertedIndex(corpus)

profiler.disable()

stats = pstats.Stats(profiler)
stats.sort_stats("cumtime")
stats.print_stats(20)