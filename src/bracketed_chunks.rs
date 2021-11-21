enum State {
    Start,
    Passthrough,
    NewShard,
    Finish,
    Exhausted
}

pub trait Sharder<T: ?Sized>: Sized {
    fn shard(&self, subj: &T) -> Option<usize>;
    fn shard_start(&self, shard: usize) -> Self;
    fn shard_end(&self, shard: usize) -> Self;
}

pub struct BracketedChunks<I, S>
  where I: Iterator, S: Sharder<I::Item>
{
    state: State,
    sharder: S,
    candidate: Option<I::Item>,
    source: I,
}

impl<I, S> Iterator for BracketedChunks<I, S>
where
    I: Iterator,
    I::Item: Copy,
    S: Sharder<I::Item>
{
    type Item = I::Item;

    fn next(&mut self) -> Option<Self::Item> {
        let (state, cur, next) = match (&self.state, &self.candidate) {
            (State::Start, None) => {
                let first = self.source.find(|v| v.shard().is_some()).unwrap();
                (State::Passthrough, Some(Sharder::shard_start(first.shard().unwrap())), Some(first))
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
                        (State::Finish, self.candidate, Some(Sharder::shard_end(shard)))
                },
                // Cur is valid, but no next
                (Some(shard), None) =>
                    (State::Finish, self.candidate, Some(Sharder::shard_end(shard))),
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

pub trait Bracketed<S>: Iterator
  where S: Sharder<Self::Item>
{
    fn bracketed_chunks(self, sharder: S) -> BracketedChunks<Self, S>
        where Self: Sized, S: Sized
    {
        BracketedChunks {
            state: State::Start,
            sharder: sharder,
            candidate: None,
            source: self
        }
    }
}

impl<I: Iterator, S> Bracketed<S> for I
    where S: Sharder<I::Item> {}

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
        struct IntSharder {
            min: u32,
            max: u32
        }
        impl Sharder<u32> for IntSharder {
            fn shard(&self, val: &u32) -> Option<usize> {
                match val {
                    v if *v < self.min => None,
                    v if *v > self.max => None,
                    v => Some(v/10)
                }
            }
            fn shard_start(&self, shard: usize) -> u32 { shard*10 }
            fn shard_end(&self, shard: usize) -> u32 { (shard+1)*10 }
        }

        let bracketed = dbgIter((5..50).step_by(10).bracketed_chunks());
        assert!(itertools::equal([10,15,20,25,30,35,40], bracketed));
    }
}
