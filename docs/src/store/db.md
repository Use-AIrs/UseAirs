# MangoDB
`mangodb.rs`, operates [MangoDB](https://www.mongodb.com/docs/drivers/rust/current/) in synchronous.
Since we expect this Db to run on a system what also handles our data in parallel on the cpu and
since we don't have to worry about huge db requests it is decided that we only communicate
synchronised with the Db so we don't spawn threads that block building threads of performance critical features.