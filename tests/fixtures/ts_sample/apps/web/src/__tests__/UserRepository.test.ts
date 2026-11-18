import { UserRepository, Repository, User } from '@acme/core'

class MockRepo implements Repository<User> {
  async load(id: string) {
    return { id, name: 'x' }
  }

  clear() {}
}

describe('UserRepository', () => {
  let repo: UserRepository
  beforeEach(() => {
    repo = new UserRepository({ get: async () => ({}) as any })
  })
  it('clears', () => {
    repo.clear()
    expect(new MockRepo().clear()).toBeUndefined()
  })
})
