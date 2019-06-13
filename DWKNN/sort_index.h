#pragma once
/* for sorting index (create sample data) */
#include<random>
#include<numeric>

template<class Ran>
class IndexSortFunctor {
public:
	explicit IndexSortFunctor(const Ran iter_begin_, const Ran iter_end_) : itbg(iter_begin_), ited(iter_end_) {}
	~IndexSortFunctor() {}
	bool operator()(const size_t a, const size_t b) const {
		return *(itbg + a) < *(itbg + b);
	}

protected:
private:
	const Ran itbg;
	const Ran ited;
};

template<class Ran>
inline IndexSortFunctor<Ran> IndexSortCmp(const Ran iter_begin_, const Ran iter_end_)
{
	return IndexSortFunctor<Ran>(iter_begin_, iter_end_);
}

/* generate Unique Random Integers*/
static std::vector<int> generateUniqueRandomIntegers(size_t max, int num)
{
	int rnd = 0;		//seed 
	std::mt19937_64 mt(rnd);
	std::vector<int> v(max + 1); // CAUTION "max+1" !!
	std::iota(v.begin(), v.end(), 0);
	std::shuffle(v.begin(), v.end(), mt);
	v.erase(v.begin() + num, v.end());
	return v;
}