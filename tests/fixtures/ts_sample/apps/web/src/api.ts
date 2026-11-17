import { UserRepository, createClient, User } from '@acme/core'

export const repository = new UserRepository(createClient())

export const userApi = {
  async fetchUser(id: string): Promise<User> {
    return repository.load(id)
  },
  reset: () => repository.clear(),
}
