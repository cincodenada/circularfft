enum State {
    Start,
    Passthrough,
    NewShard,
    Finish,
    Exhausted
}

pub trait Shardable {
    fn shard(&self) -> Option<usize>;
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
        let (state, cur, next) = match (&self.state, &self.candidate) {
            (State::Start, None) => {
                let first = self.source.find(|v| v.shard().is_some()).unwrap();
                (State::Passthrough, Some(Shardable::shard_start(first.shard().unwrap())), Some(first))
            },
            (State::Start, Some(_)) =>
                panic!("Invalid state, Start with candidate!"),
            (State::Passthrough, Some(c)) => match (c.shard(), self.source.next()) {
                // Cur is valid, and we have a next
                (Some(shard), Some(next)) => match next.shard() {
                    Some(next_shard) if shard == next_shard =>
                        (State::Passthrough, self.candidate, Some(next)),
                    Some(_) =>
                        (State::NewShard, self.candidate, Some(next)),
                    None => 
                        (State::Finish, self.candidate, Some(Shardable::shard_end(shard)))
                },
                // Cur is valid, but no next
                (Some(shard), None) =>
                    (State::Finish, self.candidate, Some(Shardable::shard_end(shard))),
                // Cur is invalid
                (None, _) =>
                    panic!("Invalid state, passthrough with out-of-range value!")
            },
            (State::Passthrough, None) =>
                panic!("Invalid state, passthrough without candidate!"),
            (State::NewShard, Some(c)) => 
                (State::Passthrough, Some(Self::Item::shard_start(c.shard().unwrap())), self.candidate),
            (State::NewShard, None) =>
                panic!("Invalid state, new shard without candidate!"),
            (State::Finish, _) => 
                (State::Exhausted, self.candidate, None),
            (State::Exhausted, _) =>
                (State::Exhausted, None, None)
        };
        self.candidate = next;
        self.state = state;
        cur
    }
}

pub trait Bracketed: Iterator {
    fn bracketed_chunks(self) -> BracketedChunks<Self>
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
            fn shard(&self) -> Option<usize> {
                match self {
                    v if *v < 10 => None,
                    v if *v > 40 => None,
                    v => Some(v/10)
                }
            }
            fn shard_start(shard: usize) -> usize { shard*10 }
            fn shard_end(shard: usize) -> usize { (shard+1)*10 }
        }

        let bracketed = dbgIter((5..50).step_by(10).bracketed_chunks());
        assert!(itertools::equal([10,15,20,25,30,35,40], bracketed));
    }
}
