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

pub struct BracketedChunks<I, R>
  where I: Iterator, R: PartialOrd<I::Item>
{
    state: State,
    min: R,
    max: R,
    candidate: Option<I::Item>,
    source: I,
}

impl<I, R> Iterator for BracketedChunks<I, R>
where
    I: Iterator,
    I::Item: Shardable + Copy,
    R: PartialOrd<I::Item>
{
    type Item = I::Item;

    fn next(&mut self) -> Option<Self::Item> {
        let (cur, next, state) = match (&self.state, &self.candidate) {
            (State::Start, None) => {
                let min = self.min;
                (Some(min), self.source.find(|v| min < *v), State::Passthrough)
            },
            (State::Start, _) =>
                panic!("Invalid state, had candidate at start!"),
            (State::Passthrough, Some(c)) => match self.source.next() {
                Some(next) if next >= self.max =>
                    (self.candidate, Some(self.max), State::Finish),
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

pub trait Bracketed: Iterator {
    fn bracketed_chunks<T>(self, min: T, max: T) -> BracketedChunks<Self, T>
        where Self:Sized, T: PartialOrd<Self::Item> {
        BracketedChunks {
            min, max,
            state: State::Start,
            candidate: None,
            source: self
        }
    }
}

impl<I: Iterator> Bracketed for I {}

#[cfg(test)]
mod tests {
    use super::*;
    use itertools;

    fn dbgIter<I, T>(it: I) -> impl Iterator<Item=T> where I: Iterator<Item=T>, T: std::fmt::Debug {
        let collected = it.collect::<Vec<_>>();
        dbg!(&collected);
        collected.into_iter()
    }

    #[test]
    fn partitions_things() {
        impl Shardable for usize {
            fn shard(&self) -> usize {
                self/10 as usize
            }
            fn from_shard(shard: usize) -> usize { shard*10 }
        }

        let bracketed = dbgIter((5..50).step_by(10).bracketed_chunks(10, 40));
        assert!(itertools::equal([10,15,20,25,30,35,40], bracketed));
    }
}
