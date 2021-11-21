enum State {
    Start,
    Passthrough,
    NewShard,
    Finish,
    Exhausted
}

pub trait Shardable {
    fn shard(&self) -> Some(usize);
    fn shard_start(shard: usize) -> Self;
    fn shard_end(shard: usize) -> Self;
}

pub struct BracketedChunks<I> where I: Iterator
{
    state: State,
    candidate: Option<I::Item>,
    source: I,
}

impl<I> Iterator for BracketedChunks<I>
where
    I: Iterator,
    I::Item: Shardable + Copy
{
    type Item = I::Item;

    fn next(&mut self) -> Option<Self::Item> {
        let candidate_shard = c.shard();
        let (state, cur, next) = match (&self.state, &self.candidate, candidate_shard) {
            (State::Start, c, None) => {
                (State::Start, None, None)
                (Shardable::shard_start(first.shard()), first, State::Passthrough)
            },
            (State::Start, c, Some(shard)) =>
                (State::Passthrough, Shardable::shard_start(shard), c)
            (State::Passthrough, Some(c), Some(shard)) => match self.source.next() {
                Some(next) if next >= self.max =>
                    (self.candidate, Shardable::shard_end(shard), State::Finish),
                Some(next) if next.shard() == next.shard() =>
                    (State::Passthrough, self.candidate, Some(next)),
                Some(next) =>
                    (State::NewShard, self.candidate, Some(next)),
                // TODO: This assumes last is always the end of the last shard
                // which is probably true for my case but could be more general
                None => (State::Finish, self.candidate, Shardable::shard_end(shard))
            },
            (State::Passthrough, None) =>
                panic!("Invalid state, passthrough without candidate!"),
            (State::NewShard, Some(c)) => 
                (STate::Passthrough, Some(Self::Item::shard_start(c.shard())), self.candidate),
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
    fn bracketed_chunks<T>(self) -> BracketedChunks<Self, T>
        where Self:Sized
    {
        BracketedChunks {
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
            fn shard_start(shard: usize) -> usize { shard*10 }
            fn shard_end(shard: usize) -> usize { (shard+1)*10 }
        }

        let bracketed = dbgIter((5..50).step_by(10).bracketed_chunks(10, 40));
        assert!(itertools::equal([10,15,20,20,25,30,30,35,40], bracketed));
    }
}
