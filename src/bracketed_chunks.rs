use std::cmp::PartialOrd;

enum State {
    Start,
    Passthrough,
    NewShard,
    Finish,
    Exhausted
}

pub trait Shardable {
    fn shard(&self) -> usize;
    fn from_shard(shard: usize) -> Self;
    
    fn same_shard(&self, other: &Self) -> bool {
        self.shard() == other.shard()
    }
}

pub struct BracketedChunks<I> where I: Iterator
{
    state: State,
    min: I::Item,
    max: I::Item,
    candidate: Option<I::Item>,
    source: I,
}

impl<I> Iterator for BracketedChunks<I>
where
    I: Iterator,
    I::Item: Shardable + PartialOrd + Copy
{
    type Item = I::Item;

    fn next(&mut self) -> Option<Self::Item> {
        let (cur, next, state) = match (&self.state, &self.candidate) {
            (State::Start, None) => {
                let min = self.min;
                (Some(min), self.source.find(|v| *v >= min), State::Passthrough)
            },
            (State::Start, _) =>
                panic!("Invalid state, had candidate at start!"),
            (State::Passthrough, Some(c)) => match self.source.next() {
                Some(next) if next.same_shard(&c) =>
                    (self.candidate, Some(next), State::Passthrough),
                Some(next) =>
                    (self.candidate, Some(next), State::NewShard),
                // TODO: This assumes max is always the end of the last shard
                // which is probably true for my case but could be more general
                None => (self.candidate, Some(self.max), State::Finish)
            },
            (State::Passthrough, None) =>
                panic!("Invalid state, passthrough without candidate!"),
            (State::NewShard, Some(c)) => 
                (Some(Self::Item::from_shard(c.shard())), self.candidate, State::Passthrough),
            (State::NewShard, None) =>
                panic!("Invalid state, new shard without candidate!"),
            (State::Finish, _) => (self.candidate, None, State::Exhausted),
            (State::Exhausted, _) => (None, None, State::Exhausted)
        };
        self.candidate = next;
        self.state = state;
        cur
    }
}

trait Bracketed: Iterator {
    fn bracketed_chunks(self, min: Self::Item, max: Self::Item) -> BracketedChunks<Self> where Self:Sized {
        BracketedChunks {
            min, max,
            state: State::Start,
            candidate: None,
            source: self
        }
    }
}

impl<I: Iterator> Bracketed for I {}
